"""PJM East load forecasting with separate models per hour of day."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.load_forecasting import (
    DATA_PATH,
    OUT_DIR,
    evaluate,
    fit_classical_models,
    print_table,
)
from qrc_engine import Reservoir
from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

FloatArray = NDArray[np.float64]

HOURS = tuple(range(24))
HOUR_WASHOUT = 24
HOUR_QRC_TRAIN = 1000
HOUR_QRC_TEST = 180

plt.style.use("seaborn-v0_8-whitegrid")


def prepare_hourly_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load PJME and prepare train/test dataframes with engineered features."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Place PJME_hourly.csv under the data/ directory."
        )

    dataframe = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    dataframe = dataframe.sort_values("Datetime").reset_index(drop=True)
    dataframe = dataframe.rename(columns={"PJME_MW": "load"})
    dataframe = dataframe.dropna()

    dataframe["hour"] = dataframe["Datetime"].dt.hour
    dataframe["dow"] = dataframe["Datetime"].dt.dayofweek
    dataframe["month"] = dataframe["Datetime"].dt.month
    dataframe["day_of_year"] = dataframe["Datetime"].dt.dayofyear
    dataframe["load_lag_24"] = dataframe["load"].shift(24)
    dataframe["load_lag_168"] = dataframe["load"].shift(168)
    dataframe = dataframe.dropna().reset_index(drop=True)

    train = dataframe[dataframe["Datetime"] < "2018-01-01"].copy()
    test = dataframe[dataframe["Datetime"] >= "2018-01-01"].copy()
    feature_cols = ["hour", "dow", "month", "day_of_year", "load_lag_24", "load_lag_168"]
    return train, test, feature_cols


def scale_hour_subset(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit a scaler on one hour slice and return scaled arrays."""

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_frame[feature_cols].to_numpy(dtype=float))
    x_test = scaler.transform(test_frame[feature_cols].to_numpy(dtype=float))
    y_train = train_frame["load"].to_numpy(dtype=float)
    y_test = test_frame["load"].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test


def merge_predictions(
    prediction_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-hour prediction frames back into chronological order."""

    merged = pd.concat(prediction_frames, ignore_index=True)
    merged = merged.sort_values("Datetime").reset_index(drop=True)
    return merged


def fit_full_hourly_classical(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, float | str]]]:
    """Train 24 classical expert models and merge their full-year predictions."""

    prediction_frames: list[pd.DataFrame] = []
    for hour in HOURS:
        train_hour = train[train["hour"] == hour].copy()
        test_hour = test[test["hour"] == hour].copy()
        x_train, y_train, x_test, y_test = scale_hour_subset(train_hour, test_hour, feature_cols)
        predictions = fit_classical_models(x_train, y_train, x_test)
        prediction_frame = pd.DataFrame(
            {
                "Datetime": test_hour["Datetime"].to_numpy(),
                "y_true": y_test,
                **predictions,
            }
        )
        prediction_frames.append(prediction_frame)

    merged = merge_predictions(prediction_frames)
    results = [
        evaluate("Ridge (full by hour)", merged["y_true"].to_numpy(dtype=float), merged["Ridge"].to_numpy(dtype=float)),
        evaluate(
            "Random Forest (full by hour)",
            merged["y_true"].to_numpy(dtype=float),
            merged["Random Forest"].to_numpy(dtype=float),
        ),
        evaluate("MLP (full by hour)", merged["y_true"].to_numpy(dtype=float), merged["MLP"].to_numpy(dtype=float)),
        evaluate(
            "GAM-like (full by hour)",
            merged["y_true"].to_numpy(dtype=float),
            merged["GAM-like"].to_numpy(dtype=float),
        ),
    ]
    return merged, results


def fit_hourly_experts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[dict[str, float | str]], dict[str, float]]:
    """Train per-hour classical and QRC experts on matched windows."""

    prediction_frames: list[pd.DataFrame] = []
    timings = {
        "QRC Qiskit (ridge)": 0.0,
        "QRC Qiskit (RF readout)": 0.0,
        "QRC Dynamiqs": 0.0,
        "QRC Perceval Fock+FB": 0.0,
    }

    for hour in HOURS:
        train_hour = train[train["hour"] == hour].copy().tail(HOUR_QRC_TRAIN)
        test_hour = test[test["hour"] == hour].copy().head(HOUR_QRC_TEST)
        x_train, y_train, x_test, y_test = scale_hour_subset(train_hour, test_hour, feature_cols)
        classical_predictions = fit_classical_models(x_train, y_train, x_test)
        qrc_predictions, qrc_timings = fit_qrc_models_by_hour(x_train, y_train, x_test)
        for label, seconds in qrc_timings.items():
            timings[label] += seconds

        aligned_slice = slice(HOUR_WASHOUT, None)
        prediction_frame = pd.DataFrame(
            {
                "Datetime": test_hour["Datetime"].to_numpy()[aligned_slice],
                "y_true": y_test[aligned_slice],
                "Ridge": classical_predictions["Ridge"][aligned_slice],
                "Random Forest": classical_predictions["Random Forest"][aligned_slice],
                "MLP": classical_predictions["MLP"][aligned_slice],
                "GAM-like": classical_predictions["GAM-like"][aligned_slice],
                "QRC Qiskit (ridge)": qrc_predictions["QRC Qiskit (ridge)"],
                "QRC Qiskit (RF readout)": qrc_predictions["QRC Qiskit (RF readout)"],
                "QRC Dynamiqs": qrc_predictions["QRC Dynamiqs"],
                "QRC Perceval Fock+FB": qrc_predictions["QRC Perceval Fock+FB"],
            }
        )
        prediction_frames.append(prediction_frame)

    merged = merge_predictions(prediction_frames)
    results = [
        evaluate("Ridge", merged["y_true"].to_numpy(dtype=float), merged["Ridge"].to_numpy(dtype=float)),
        evaluate(
            "Random Forest",
            merged["y_true"].to_numpy(dtype=float),
            merged["Random Forest"].to_numpy(dtype=float),
        ),
        evaluate("MLP", merged["y_true"].to_numpy(dtype=float), merged["MLP"].to_numpy(dtype=float)),
        evaluate("GAM-like", merged["y_true"].to_numpy(dtype=float), merged["GAM-like"].to_numpy(dtype=float)),
        evaluate(
            "QRC Qiskit (ridge)",
            merged["y_true"].to_numpy(dtype=float),
            merged["QRC Qiskit (ridge)"].to_numpy(dtype=float),
        ),
        evaluate(
            "QRC Qiskit (RF readout)",
            merged["y_true"].to_numpy(dtype=float),
            merged["QRC Qiskit (RF readout)"].to_numpy(dtype=float),
        ),
        evaluate(
            "QRC Dynamiqs",
            merged["y_true"].to_numpy(dtype=float),
            merged["QRC Dynamiqs"].to_numpy(dtype=float),
        ),
        evaluate(
            "QRC Perceval Fock+FB",
            merged["y_true"].to_numpy(dtype=float),
            merged["QRC Perceval Fock+FB"].to_numpy(dtype=float),
        ),
    ]
    return merged, results, timings


def fit_qrc_models_by_hour(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
) -> tuple[dict[str, FloatArray], dict[str, float]]:
    """Train the QRC configurations with the by-hour washout."""

    predictions: dict[str, FloatArray] = {}
    timings: dict[str, float] = {}

    print("\nTraining QRC models...")

    start = time.perf_counter()
    qiskit_reservoir = Reservoir(
        backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
        washout=HOUR_WASHOUT,
        alpha=1e-3,
    )
    qiskit_reservoir.fit(x_train, y_train)
    predictions["QRC Qiskit (ridge)"] = qiskit_reservoir.predict(x_test)
    timings["QRC Qiskit (ridge)"] = time.perf_counter() - start

    start = time.perf_counter()
    qiskit_rf_reservoir = Reservoir(
        backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
        washout=HOUR_WASHOUT,
        readout="random_forest",
        n_estimators=100,
        max_depth=10,
    )
    qiskit_rf_reservoir.fit(x_train, y_train)
    predictions["QRC Qiskit (RF readout)"] = qiskit_rf_reservoir.predict(x_test)
    timings["QRC Qiskit (RF readout)"] = time.perf_counter() - start

    start = time.perf_counter()
    dynamiqs_reservoir = Reservoir(
        backend=DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3),
        washout=HOUR_WASHOUT,
        alpha=1e-5,
    )
    dynamiqs_reservoir.fit(x_train, y_train)
    predictions["QRC Dynamiqs"] = dynamiqs_reservoir.predict(x_test)
    timings["QRC Dynamiqs"] = time.perf_counter() - start

    start = time.perf_counter()
    perceval_reservoir = Reservoir(
        backend=PercevalBackend(
            n_modes=5,
            n_photons=2,
            depth=2,
            fock_mode=True,
            feedback=True,
            memory_decay=0.7,
            seed=3,
        ),
        washout=HOUR_WASHOUT,
        alpha=1e-3,
    )
    perceval_reservoir.fit(x_train, y_train)
    predictions["QRC Perceval Fock+FB"] = perceval_reservoir.predict(x_test)
    timings["QRC Perceval Fock+FB"] = time.perf_counter() - start

    return predictions, timings


def save_hourly_plots(
    merged: pd.DataFrame,
    sub_results: list[dict[str, float | str]],
    full_results: list[dict[str, float | str]],
    save: bool = False,
) -> None:
    """Create the same plot family for the by-hour experiment."""

    y_true = merged["y_true"].to_numpy(dtype=float)
    classical_names = ["Ridge", "Random Forest", "MLP", "GAM-like"]
    qrc_names = [
        "QRC Qiskit (ridge)",
        "QRC Qiskit (RF readout)",
        "QRC Dynamiqs",
        "QRC Perceval Fock+FB",
    ]
    best_classical = min(
        (result for result in sub_results if str(result["name"]) in classical_names),
        key=lambda result: float(result["RMSE (MW)"]),
    )
    best_qrc = min(
        (result for result in sub_results if str(result["name"]) in qrc_names),
        key=lambda result: float(result["RMSE (MW)"]),
    )
    n_plot = 500

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axis = axes[0]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(merged["Random Forest"].to_numpy(dtype=float)[:n_plot], label="Random Forest", lw=0.9, color="#e63946")
    axis.plot(merged["MLP"].to_numpy(dtype=float)[:n_plot], label="MLP", lw=0.9, color="#457b9d")
    axis.set_title("Classical baselines (by hour)")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")

    axis = axes[1]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(merged["QRC Qiskit (ridge)"].to_numpy(dtype=float)[:n_plot], label="QRC Qiskit", lw=0.9, color="#1d3557")
    axis.plot(merged["QRC Dynamiqs"].to_numpy(dtype=float)[:n_plot], label="QRC Dynamiqs", lw=0.9, color="#2a9d8f")
    axis.plot(
        merged["QRC Perceval Fock+FB"].to_numpy(dtype=float)[:n_plot],
        label="QRC Perceval Fock+FB",
        lw=0.9,
        color="#9b5de5",
    )
    axis.set_title("Quantum reservoir models (by hour)")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")

    axis = axes[2]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(
        merged[str(best_classical["name"])].to_numpy(dtype=float)[:n_plot],
        label=str(best_classical["name"]),
        lw=0.9,
        color="#e63946",
    )
    axis.plot(
        merged[str(best_qrc["name"])].to_numpy(dtype=float)[:n_plot],
        label=str(best_qrc["name"]),
        lw=0.9,
        color="#1d3557",
    )
    axis.set_title("Best classical (lowest RMSE) vs best QRC (lowest RMSE)")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")
    axis.set_xlabel("Merged chronological test steps")

    figure.suptitle("PJM East Hourly Load Forecasting by Hour-of-Day Experts", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_by_hour_comparison.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_by_hour_comparison.png", bbox_inches="tight", dpi=150)
        plt.close(figure)

    names = [str(result["name"]) for result in sub_results]
    rmses = [float(result["RMSE (MW)"]) for result in sub_results]
    colors = ["#adb5bd", "#e63946", "#457b9d", "#e9c46a", "#1d3557", "#264653", "#2a9d8f", "#9b5de5"]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.barh(range(len(names)), rmses, color=colors, height=0.55)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=9)
    axis.set_xlabel("RMSE (MW)")
    axis.set_xlim(0, max(rmses) * 1.15)
    for bar, value in zip(bars, rmses):
        axis.text(
            bar.get_width() + (max(rmses) * 0.01),
            bar.get_y() + (bar.get_height() / 2.0),
            f"{value:.0f}",
            va="center",
            fontsize=8.5,
        )
    axis.set_title("PJM East Load Forecasting by Hour: RMSE (MW)")
    axis.invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_by_hour_rmse.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_by_hour_rmse.png", bbox_inches="tight", dpi=150)
        plt.close(figure)

    full_names = [str(result["name"]) for result in full_results]
    full_rmses = [float(result["RMSE (MW)"]) for result in full_results]
    all_names = full_names + qrc_names
    all_rmses = full_rmses + [float(result["RMSE (MW)"]) for result in sub_results[-4:]]
    all_colors = ["#dee2e6", "#c1121f", "#2b6cb0", "#d4a017", "#1d3557", "#264653", "#2a9d8f", "#9b5de5"]

    figure, axis = plt.subplots(figsize=(12, 4))
    bars = axis.barh(range(len(all_names)), all_rmses, color=all_colors, height=0.55)
    axis.set_yticks(range(len(all_names)))
    axis.set_yticklabels(all_names, fontsize=8.5)
    axis.set_xlabel("RMSE (MW)")
    axis.set_xlim(0, max(all_rmses) * 1.15)
    for bar, value in zip(bars, all_rmses):
        axis.text(
            bar.get_width() + (max(all_rmses) * 0.01),
            bar.get_y() + (bar.get_height() / 2.0),
            f"{value:.0f}",
            va="center",
            fontsize=8,
        )
    axis.axvline(
        x=full_rmses[1],
        color="#c1121f",
        ls="--",
        lw=0.7,
        alpha=0.5,
        label="RF full-data by-hour reference",
    )
    axis.legend(fontsize=8)
    axis.set_title("Full context: by-hour classical (full data) vs by-hour QRC")
    axis.invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_by_hour_full_context.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_by_hour_full_context.png", bbox_inches="tight", dpi=150)
        plt.close(figure)


def main() -> None:
    """Run the by-hour expert experiment."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test, feature_cols = prepare_hourly_dataframe()
    print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")
    print(
        f"By-hour QRC setup: {len(HOURS)} hour buckets, "
        f"{HOUR_QRC_TRAIN} train samples/hour, {HOUR_QRC_TEST} test samples/hour, "
        f"washout={HOUR_WASHOUT}"
    )

    full_merged, full_results = fit_full_hourly_classical(train, test, feature_cols)
    print_table(full_results, "Classical by-hour experts - full 2018 test set")

    merged, sub_results, timings = fit_hourly_experts(train, test, feature_cols)
    print_table(sub_results, "By-hour experts - matched classical vs QRC comparison")

    print("\nTiming (QRC, summed across 24 hour experts):")
    for label, seconds in timings.items():
        print(f"  {label:<24} {seconds:>8.0f}s")

    save_hourly_plots(merged, sub_results, full_results)
    print(f"\nMerged aligned evaluation rows: {len(merged):,}")
    print(f"Artifacts saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
