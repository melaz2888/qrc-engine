"""PJM East hourly load forecasting: QRC backends vs classical models.

Run from the project root:
    python experiments/load_forecasting.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, SplineTransformer

from qrc_engine import Reservoir
from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "PJME_hourly.csv"
OUT_DIR = ROOT / "experiments"

WASHOUT = 100
N_QRC_TRAIN = 5000
N_QRC_TEST = 2000

plt.style.use("seaborn-v0_8-whitegrid")
warnings.filterwarnings("ignore", module="perceval.utils.persistent_data")


def evaluate(name: str, y_true: FloatArray, y_pred: FloatArray) -> dict[str, float | str]:
    """Compute the forecasting metrics for one model."""

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    nrmse = rmse / float(np.std(y_true))
    return {"name": name, "RMSE (MW)": rmse, "MAE (MW)": mae, "MAPE (%)": mape, "NRMSE": nrmse}


def print_table(results: list[dict[str, float | str]], title: str) -> None:
    """Print a formatted metric table."""

    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    print(f"  {'Model':<28} | {'RMSE (MW)':>10} | {'MAE (MW)':>10} | {'MAPE (%)':>9} | {'NRMSE':>7}")
    print(f"  {'-' * 74}")
    for result in results:
        print(
            f"  {str(result['name']):<28} | {float(result['RMSE (MW)']):>10.1f} |"
            f" {float(result['MAE (MW)']):>10.1f} | {float(result['MAPE (%)']):>9.2f} |"
            f" {float(result['NRMSE']):>7.3f}"
        )
    print(f"{'=' * 78}")


def load_dataset() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Load PJME and build the train/test features."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset at {DATA_PATH}. Place PJME_hourly.csv under the data/ directory."
        )

    print("Loading data...")
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
    target_col = "load"

    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    x_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    print(f"Train: {len(x_train):,} samples | Test: {len(x_test):,} samples")
    return x_train, y_train, x_test, y_test


def scale_features(
    x_train: FloatArray,
    x_test: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Fit a min-max scaler on train only and transform both splits."""

    scaler = MinMaxScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def fit_classical_models(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
) -> dict[str, FloatArray]:
    """Train the classical baseline models and return their predictions."""

    predictions: dict[str, FloatArray] = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train)
    predictions["Ridge"] = np.asarray(ridge.predict(x_test), dtype=float)

    forest = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    forest.fit(x_train, y_train)
    predictions["Random Forest"] = np.asarray(forest.predict(x_test), dtype=float)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    mlp.fit(x_train, y_train)
    predictions["MLP"] = np.asarray(mlp.predict(x_test), dtype=float)

    gam_like = make_pipeline(SplineTransformer(n_knots=10, degree=3), Ridge(alpha=1.0))
    gam_like.fit(x_train, y_train)
    predictions["GAM-like"] = np.asarray(gam_like.predict(x_test), dtype=float)
    return predictions


def fit_qrc_models(
    x_train: FloatArray,
    y_train: FloatArray,
    x_test: FloatArray,
) -> tuple[dict[str, FloatArray], dict[str, float]]:
    """Train the QRC configurations and return predictions plus timings."""

    predictions: dict[str, FloatArray] = {}
    timings: dict[str, float] = {}

    print("\nTraining QRC models...")

    print("  Qiskit (ridge)...", end=" ", flush=True)
    start = time.perf_counter()
    qiskit_reservoir = Reservoir(
        backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
        washout=WASHOUT,
        alpha=1e-3,
    )
    qiskit_reservoir.fit(x_train, y_train)
    predictions["QRC Qiskit (ridge)"] = qiskit_reservoir.predict(x_test)
    timings["QRC Qiskit (ridge)"] = time.perf_counter() - start
    print(f"done ({timings['QRC Qiskit (ridge)']:.0f}s)")

    print("  Qiskit (RF readout)...", end=" ", flush=True)
    start = time.perf_counter()
    qiskit_rf_reservoir = Reservoir(
        backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
        washout=WASHOUT,
        readout="random_forest",
        n_estimators=100,
        max_depth=10,
    )
    qiskit_rf_reservoir.fit(x_train, y_train)
    predictions["QRC Qiskit (RF readout)"] = qiskit_rf_reservoir.predict(x_test)
    timings["QRC Qiskit (RF readout)"] = time.perf_counter() - start
    print(f"done ({timings['QRC Qiskit (RF readout)']:.0f}s)")

    print("  Dynamiqs...", end=" ", flush=True)
    start = time.perf_counter()
    dynamiqs_reservoir = Reservoir(
        backend=DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3),
        washout=WASHOUT,
        alpha=1e-5,
    )
    dynamiqs_reservoir.fit(x_train, y_train)
    predictions["QRC Dynamiqs"] = dynamiqs_reservoir.predict(x_test)
    timings["QRC Dynamiqs"] = time.perf_counter() - start
    print(f"done ({timings['QRC Dynamiqs']:.0f}s)")

    print("  Perceval Fock+Feedback...", end=" ", flush=True)
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
        washout=WASHOUT,
        alpha=1e-3,
    )
    perceval_reservoir.fit(x_train, y_train)
    predictions["QRC Perceval Fock+FB"] = perceval_reservoir.predict(x_test)
    timings["QRC Perceval Fock+FB"] = time.perf_counter() - start
    print(f"done ({timings['QRC Perceval Fock+FB']:.0f}s)")

    return predictions, timings


def save_time_domain_plot(
    y_true: FloatArray,
    classical_predictions: dict[str, FloatArray],
    qrc_predictions: dict[str, FloatArray],
    sub_results: list[dict[str, float | str]],
    save: bool = False,
) -> None:
    """Create the main time-domain comparison figure."""

    n_plot = 500
    classical_names = set(classical_predictions.keys())
    qrc_names = set(qrc_predictions.keys())
    best_classical = min(
        (result for result in sub_results if str(result["name"]) in classical_names),
        key=lambda result: float(result["RMSE (MW)"]),
    )
    best_qrc = min(
        (result for result in sub_results if str(result["name"]) in qrc_names),
        key=lambda result: float(result["RMSE (MW)"]),
    )
    best_classical_name = str(best_classical["name"])
    best_qrc_name = str(best_qrc["name"])
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axis = axes[0]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(classical_predictions["Random Forest"][:n_plot], label="Random Forest", lw=0.9, color="#e63946")
    axis.plot(classical_predictions["MLP"][:n_plot], label="MLP", lw=0.9, color="#457b9d")
    axis.set_title("Classical baselines (subsample)")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")

    axis = axes[1]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(qrc_predictions["QRC Qiskit (ridge)"][:n_plot], label="QRC Qiskit", lw=0.9, color="#1d3557")
    axis.plot(qrc_predictions["QRC Dynamiqs"][:n_plot], label="QRC Dynamiqs", lw=0.9, color="#2a9d8f")
    axis.plot(qrc_predictions["QRC Perceval Fock+FB"][:n_plot], label="QRC Perceval Fock+FB", lw=0.9, color="#9b5de5")
    axis.set_title("Quantum reservoir models")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")

    axis = axes[2]
    axis.plot(y_true[:n_plot], label="Ground truth", lw=1.5, color="#adb5bd")
    axis.plot(
        classical_predictions[best_classical_name][:n_plot],
        label=best_classical_name,
        lw=0.9,
        color="#e63946",
    )
    axis.plot(
        qrc_predictions[best_qrc_name][:n_plot],
        label=best_qrc_name,
        lw=0.9,
        color="#1d3557",
    )
    axis.set_title("Best classical (lowest RMSE) vs best QRC (lowest RMSE)")
    axis.legend(fontsize=8)
    axis.set_ylabel("MW")
    axis.set_xlabel("Hour")

    figure.suptitle("PJM East Hourly Load Forecasting - 2018 test set", y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_comparison.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_comparison.png", bbox_inches="tight", dpi=150)
        plt.close(figure)
        print("Saved load_forecast_comparison.pdf")


def save_rmse_plot(sub_results: list[dict[str, float | str]], save: bool = False) -> None:
    """Create the RMSE bar chart."""

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
    axis.set_title("PJM East Load Forecasting: RMSE (MW) - same training window")
    axis.invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_rmse.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_rmse.png", bbox_inches="tight", dpi=150)
        plt.close(figure)
        print("Saved load_forecast_rmse.pdf")


def save_context_plot(
    full_results: list[dict[str, float | str]],
    sub_results: list[dict[str, float | str]],
    save: bool = False,
) -> None:
    """Create the full-context classical-vs-QRC comparison plot."""

    full_names = [str(result["name"]) for result in full_results]
    full_rmses = [float(result["RMSE (MW)"]) for result in full_results]
    qrc_names = [
        "QRC Qiskit (ridge)",
        "QRC Qiskit (RF readout)",
        "QRC Dynamiqs",
        "QRC Perceval Fock+FB",
    ]
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
        label="RF full-data reference",
    )
    axis.legend(fontsize=8)
    axis.set_title("Full context: classical (full data) vs QRC (5,000 samples)")
    axis.invert_yaxis()
    plt.tight_layout()
    if save:
        plt.savefig(OUT_DIR / "load_forecast_full_context.pdf", bbox_inches="tight", dpi=150)
        plt.savefig(OUT_DIR / "load_forecast_full_context.png", bbox_inches="tight", dpi=150)
        plt.close(figure)
        print("Saved load_forecast_full_context.pdf")


def main() -> None:
    """Run the full classical-vs-QRC load forecasting benchmark."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_dataset()
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)

    print("\nTraining classical models on full dataset...")
    classical_full = fit_classical_models(x_train_scaled, y_train, x_test_scaled)
    full_results = [
        evaluate("Ridge (full data)", y_test, classical_full["Ridge"]),
        evaluate("Random Forest (full data)", y_test, classical_full["Random Forest"]),
        evaluate("MLP (full data)", y_test, classical_full["MLP"]),
        evaluate("GAM-like (full data)", y_test, classical_full["GAM-like"]),
    ]
    print_table(full_results, "Classical models - full training set")

    x_qrc_train = x_train_scaled[-N_QRC_TRAIN:]
    y_qrc_train = y_train[-N_QRC_TRAIN:]
    x_qrc_test = x_test_scaled[:N_QRC_TEST]
    y_qrc_test = y_test[:N_QRC_TEST]
    y_aligned = y_qrc_test[WASHOUT:]

    print(
        f"\nQRC subsample: {N_QRC_TRAIN:,} train / {N_QRC_TEST:,} test "
        f"(aligned test: {len(y_aligned):,} samples)"
    )

    print("\nRetraining classical models on QRC subsample...")
    classical_sub = fit_classical_models(x_qrc_train, y_qrc_train, x_qrc_test)
    aligned_classical = {
        name: prediction[WASHOUT:]
        for name, prediction in classical_sub.items()
    }

    qrc_predictions, qrc_timings = fit_qrc_models(x_qrc_train, y_qrc_train, x_qrc_test)

    sub_results = [
        evaluate("Ridge", y_aligned, aligned_classical["Ridge"]),
        evaluate("Random Forest", y_aligned, aligned_classical["Random Forest"]),
        evaluate("MLP", y_aligned, aligned_classical["MLP"]),
        evaluate("GAM-like", y_aligned, aligned_classical["GAM-like"]),
        evaluate("QRC Qiskit (ridge)", y_aligned, qrc_predictions["QRC Qiskit (ridge)"]),
        evaluate("QRC Qiskit (RF readout)", y_aligned, qrc_predictions["QRC Qiskit (RF readout)"]),
        evaluate("QRC Dynamiqs", y_aligned, qrc_predictions["QRC Dynamiqs"]),
        evaluate("QRC Perceval Fock+FB", y_aligned, qrc_predictions["QRC Perceval Fock+FB"]),
    ]
    print_table(sub_results, f"Fair comparison - same {N_QRC_TRAIN:,}-sample window")

    print("\nTiming (QRC):")
    for label, seconds in qrc_timings.items():
        print(f"  {label:<24} {seconds:>8.0f}s")

    save_time_domain_plot(y_aligned, aligned_classical, qrc_predictions, sub_results)
    save_rmse_plot(sub_results)
    save_context_plot(full_results, sub_results)


if __name__ == "__main__":
    main()
