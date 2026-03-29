"""Generate the qrc-engine evaluation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    """Write the evaluation notebook to disk."""

    notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "qrc_engine_evaluation.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        }
    )

    cells = [
        # ── Title ──────────────────────────────────────────────────────────
        nbf.v4.new_markdown_cell(
            "# qrc-engine Evaluation\n\n"
            "Full benchmark of all backends and all evolution modes — including the new "
            "**projective measurement / wavefunction collapse** features — on three tasks:\n\n"
            "- **Mackey-Glass** (chaotic 1-D forecasting)\n"
            "- **NARMA-10** (nonlinear memory, order 10)\n"
            "- **Lorenz** (3-D multivariate input)\n\n"
            "Metric: **NRMSE** (lower is better; > 1.0 = worse than predicting the mean)."
        ),

        # ── Cell 1: Setup ──────────────────────────────────────────────────
        nbf.v4.new_code_cell(
            "import warnings\n"
            "import time\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "from qrc_engine import Reservoir\n"
            "from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend\n"
            "from qrc_engine.tasks import lorenz_system, mackey_glass, narma10\n"
            "from qrc_engine.utils import nrmse\n\n"
            "warnings.filterwarnings('ignore')\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "WASHOUT = 50\n\n"
            "def run_config(label, backend, alpha, X_tr, y_tr, X_te, y_te):\n"
            "    t0 = time.perf_counter()\n"
            "    res = Reservoir(backend=backend, washout=WASHOUT, alpha=alpha)\n"
            "    res.fit(X_tr, y_tr)\n"
            "    pred = res.predict(X_te)\n"
            "    elapsed = time.perf_counter() - t0\n"
            "    score = nrmse(y_te[WASHOUT:], pred)\n"
            "    return label, score, pred, elapsed"
        ),

        # ── Cell 2: Load tasks ─────────────────────────────────────────────
        nbf.v4.new_code_cell(
            "X_mg_tr, y_mg_tr, X_mg_te, y_mg_te = mackey_glass(n_samples=2000, split=0.8, seed=11)\n"
            "X_nm_tr, y_nm_tr, X_nm_te, y_nm_te = narma10(n_samples=2000, split=0.8, seed=11)\n"
            "X_lz_tr, y_lz_tr, X_lz_te, y_lz_te = lorenz_system(n_samples=2000, split=0.8, seed=7)\n"
            "print('Tasks loaded.')"
        ),

        # ── Cell 3: Qiskit — Mackey-Glass ──────────────────────────────────
        nbf.v4.new_markdown_cell(
            "## QiskitBackend — All Modes on Mackey-Glass\n\n"
            "Four modes: stateless rebuild, persistent statevector, "
            "persistent + **full projective collapse**, persistent + **partial collapse** (2 of 4 qubits measured)."
        ),
        nbf.v4.new_code_cell(
            "qk_mg = []\n"
            "qk_mg.append(run_config('Default (rebuild each step)',\n"
            "    QiskitBackend(n_qubits=4, depth=4, seed=11), 1e-3,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "qk_mg.append(run_config('Persistent (no collapse)',\n"
            "    QiskitBackend(n_qubits=4, depth=4, persistent_state=True, seed=11), 1e-3,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "qk_mg.append(run_config('Persistent + full collapse',\n"
            "    QiskitBackend(n_qubits=4, depth=4, persistent_state=True,\n"
            "                  measure_and_collapse=True, seed=11), 1e-3,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "qk_mg.append(run_config('Persistent + partial (2/4)',\n"
            "    QiskitBackend(n_qubits=4, depth=4, persistent_state=True,\n"
            "                  measure_and_collapse=True, n_measured_qubits=2, seed=11), 1e-3,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n\n"
            "for l, s, _, t in qk_mg:\n"
            "    print(f'  {l}: NRMSE={s:.4f} ({t:.1f}s)')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 4))\n"
            "ground = y_mg_te[WASHOUT:]\n"
            "ax.plot(ground[:300], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip(qk_mg, ['#1d3557', '#2a9d8f', '#e63946', '#e9c46a']):\n"
            "    ax.plot(p[:300], label=f'{l}  NRMSE={s:.3f}', lw=0.9, color=c, alpha=0.85)\n"
            "ax.set_xlabel('Time step'); ax.set_ylabel('Value')\n"
            "ax.set_title('QiskitBackend: all modes on Mackey-Glass')\n"
            "ax.legend(loc='upper right', fontsize=7)\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 4: Perceval — Mackey-Glass ────────────────────────────────
        nbf.v4.new_markdown_cell(
            "## PercevalBackend — All Modes on Mackey-Glass\n\n"
            "Four modes: classical field, Fock analytic, "
            "Fock + **threshold detection** (sampled click statistics), Fock + feedback."
        ),
        nbf.v4.new_code_cell(
            "pcv_mg = []\n"
            "pcv_mg.append(run_config('Classical field',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, memory_decay=0.95, seed=3), 1e-4,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "pcv_mg.append(run_config('Fock (analytic)',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    memory_decay=0.7, seed=3), 1e-4,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "pcv_mg.append(run_config('Fock + threshold det.',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    threshold_detection=True, memory_decay=0.7,\n"
            "                    n_detection_samples=2000, seed=3), 1e-4,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "pcv_mg.append(run_config('Fock + feedback',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    feedback=True, memory_decay=0.7, seed=3), 1e-4,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n\n"
            "for l, s, _, t in pcv_mg:\n"
            "    print(f'  {l}: NRMSE={s:.4f} ({t:.1f}s)')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 4))\n"
            "ax.plot(ground[:300], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip(pcv_mg, ['#9b5de5', '#c77dff', '#f15bb5', '#00bbf9']):\n"
            "    ax.plot(p[:300], label=f'{l}  NRMSE={s:.3f}', lw=0.9, color=c, alpha=0.85)\n"
            "ax.set_xlabel('Time step'); ax.set_ylabel('Value')\n"
            "ax.set_title('PercevalBackend: all modes on Mackey-Glass')\n"
            "ax.legend(loc='upper right', fontsize=7)\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 5: Dynamiqs — Mackey-Glass ────────────────────────────────
        nbf.v4.new_markdown_cell(
            "## DynamiqsBackend — All Modes on Mackey-Glass\n\n"
            "Three modes: convex-mixture dissipation, Lindblad master equation, "
            "mixture + **projective measurement collapse**."
        ),
        nbf.v4.new_code_cell(
            "dq_mg = []\n"
            "dq_mg.append(run_config('Convex mixture',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3), 1e-5,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "dq_mg.append(run_config('Lindblad',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, lindblad=True, seed=3), 1e-5,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n"
            "dq_mg.append(run_config('Mixture + collapse',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1,\n"
            "                    projective_measurement=True, seed=3), 1e-5,\n"
            "    X_mg_tr, y_mg_tr, X_mg_te, y_mg_te))\n\n"
            "for l, s, _, t in dq_mg:\n"
            "    print(f'  {l}: NRMSE={s:.4f} ({t:.1f}s)')\n\n"
            "fig, ax = plt.subplots(figsize=(11, 4))\n"
            "ax.plot(ground[:300], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip(dq_mg, ['#2a9d8f', '#e9c46a', '#e76f51']):\n"
            "    ax.plot(p[:300], label=f'{l}  NRMSE={s:.3f}', lw=0.9, color=c, alpha=0.85)\n"
            "ax.set_xlabel('Time step'); ax.set_ylabel('Value')\n"
            "ax.set_title('DynamiqsBackend: all modes on Mackey-Glass')\n"
            "ax.legend(loc='upper right', fontsize=7)\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 6: Mackey-Glass bar chart ─────────────────────────────────
        nbf.v4.new_markdown_cell("## Mackey-Glass Summary"),
        nbf.v4.new_code_cell(
            "all_mg = qk_mg + pcv_mg + dq_mg\n"
            "labels_mg = [l for l, s, _, _ in all_mg]\n"
            "scores_mg = [s for _, s, _, _ in all_mg]\n"
            "colors_mg = ['#1d3557', '#2a9d8f', '#e63946', '#e9c46a',\n"
            "             '#9b5de5', '#c77dff', '#f15bb5', '#00bbf9',\n"
            "             '#264653', '#e9c46a', '#e76f51']\n\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "y_pos = np.arange(len(labels_mg))\n"
            "bars = ax.barh(y_pos, scores_mg, color=colors_mg, height=0.55)\n"
            "ax.set_yticks(y_pos); ax.set_yticklabels(labels_mg, fontsize=9)\n"
            "ax.set_xlabel('NRMSE (lower is better)')\n"
            "ax.set_xlim(0, max(scores_mg) * 1.12)\n"
            "ax.axvline(x=1.0, color='#666', ls='--', lw=0.7, label='Baseline (mean)')\n"
            "for bar, score in zip(bars, scores_mg):\n"
            "    ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,\n"
            "            f'{score:.3f}', va='center', fontsize=8.5)\n"
            "ax.set_title('Mackey-Glass NRMSE: all backends, all modes')\n"
            "ax.legend(fontsize=8, loc='lower right')\n"
            "ax.invert_yaxis()\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 7: NARMA-10 ───────────────────────────────────────────────
        nbf.v4.new_markdown_cell(
            "## NARMA-10 — All Modes\n\n"
            "NARMA-10 requires retaining 10 past inputs through a nonlinear recurrence — "
            "a harder task that stresses memory depth."
        ),
        nbf.v4.new_code_cell(
            "nm = []\n"
            "nm.append(run_config('Qiskit default',\n"
            "    QiskitBackend(n_qubits=4, depth=4, seed=11), 1e-3,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Dynamiqs mixture',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3), 1e-5,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Dynamiqs Lindblad',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, lindblad=True, seed=3), 1e-5,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Dynamiqs mixture+collapse',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1,\n"
            "                    projective_measurement=True, seed=3), 1e-5,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Perceval field',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, memory_decay=0.95, seed=3), 1e-4,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Perceval Fock analytic',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    memory_decay=0.7, seed=3), 1e-1,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Perceval Fock threshold',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    threshold_detection=True, memory_decay=0.7,\n"
            "                    n_detection_samples=2000, seed=3), 1e-1,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n"
            "nm.append(run_config('Perceval Fock feedback',\n"
            "    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True,\n"
            "                    feedback=True, memory_decay=0.7, seed=3), 1e-1,\n"
            "    X_nm_tr, y_nm_tr, X_nm_te, y_nm_te))\n\n"
            "for l, s, _, t in nm:\n"
            "    print(f'  {l}: NRMSE={s:.4f} ({t:.1f}s)')\n\n"
            "fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)\n"
            "ground_nm = y_nm_te[WASHOUT:]\n"
            "n_plot = 200\n\n"
            "ax = axes[0]\n"
            "ax.plot(ground_nm[:n_plot], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip([nm[0], nm[1]], ['#1d3557', '#2a9d8f']):\n"
            "    ax.plot(p[:n_plot], label=f'{l} ({s:.3f})', lw=0.9, color=c)\n"
            "ax.set_title('Qiskit and Dynamiqs'); ax.legend(fontsize=7, loc='upper right')\n\n"
            "ax = axes[1]\n"
            "ax.plot(ground_nm[:n_plot], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip([nm[1], nm[2], nm[3]], ['#2a9d8f', '#e9c46a', '#e76f51']):\n"
            "    ax.plot(p[:n_plot], label=f'{l} ({s:.3f})', lw=0.9, color=c)\n"
            "ax.set_title('Dynamiqs modes'); ax.legend(fontsize=7, loc='upper right')\n\n"
            "ax = axes[2]\n"
            "ax.plot(ground_nm[:n_plot], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "for (l, s, p, _), c in zip([nm[4], nm[5], nm[6], nm[7]], ['#9b5de5', '#c77dff', '#f15bb5', '#00bbf9']):\n"
            "    ax.plot(p[:n_plot], label=f'{l} ({s:.3f})', lw=0.9, color=c)\n"
            "ax.set_title('Perceval modes'); ax.set_xlabel('Time step')\n"
            "ax.legend(fontsize=7, loc='upper right')\n\n"
            "fig.suptitle('NARMA-10: all backends, all modes', y=1.01)\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 8: NARMA-10 bar chart ─────────────────────────────────────
        nbf.v4.new_markdown_cell("## NARMA-10 Summary"),
        nbf.v4.new_code_cell(
            "labels_nm = [l for l, s, _, _ in nm]\n"
            "scores_nm = [s for _, s, _, _ in nm]\n"
            "colors_nm = ['#1d3557', '#2a9d8f', '#e9c46a', '#e76f51',\n"
            "             '#9b5de5', '#c77dff', '#f15bb5', '#00bbf9']\n\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "y_pos = np.arange(len(labels_nm))\n"
            "bars = ax.barh(y_pos, scores_nm, color=colors_nm, height=0.55)\n"
            "ax.set_yticks(y_pos); ax.set_yticklabels(labels_nm, fontsize=9)\n"
            "ax.set_xlabel('NRMSE (lower is better)')\n"
            "ax.set_xlim(0, max(scores_nm) * 1.12)\n"
            "ax.axvline(x=1.0, color='#666', ls='--', lw=0.7, label='Baseline (mean)')\n"
            "for bar, score in zip(bars, scores_nm):\n"
            "    ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,\n"
            "            f'{score:.3f}', va='center', fontsize=8.5)\n"
            "ax.set_title('NARMA-10 NRMSE: all backends, all modes')\n"
            "ax.legend(fontsize=8, loc='lower right')\n"
            "ax.invert_yaxis()\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 9: Lorenz ─────────────────────────────────────────────────
        nbf.v4.new_markdown_cell(
            "## Lorenz — 3D Multivariate Input\n\n"
            "The 3-component Lorenz state vector is fed at each step; the target is x(t+1). "
            "Same `Reservoir` API, no code changes vs scalar tasks."
        ),
        nbf.v4.new_code_cell(
            "lz = []\n"
            "lz.append(run_config('Qiskit (5q, d3)',\n"
            "    QiskitBackend(n_qubits=5, depth=3, seed=5), 1e-3,\n"
            "    X_lz_tr, y_lz_tr, X_lz_te, y_lz_te))\n"
            "lz.append(run_config('Dynamiqs (d=5)',\n"
            "    DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3), 1e-5,\n"
            "    X_lz_tr, y_lz_tr, X_lz_te, y_lz_te))\n\n"
            "for l, s, _, t in lz:\n"
            "    print(f'  {l}: NRMSE={s:.4f} ({t:.1f}s)')\n\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)\n"
            "ground_lz = y_lz_te[WASHOUT:]\n"
            "for ax, (l, s, p, _), c in zip(axes, lz, ['#1d3557', '#2a9d8f']):\n"
            "    ax.plot(ground_lz[:300], label='Ground truth', lw=1.5, color='#adb5bd')\n"
            "    ax.plot(p[:300], label=f'{l} ({s:.3f})', lw=0.9, color=c)\n"
            "    ax.set_ylabel('x(t+1)'); ax.legend(fontsize=8, loc='upper right')\n"
            "axes[-1].set_xlabel('Time step')\n"
            "fig.suptitle('Lorenz x(t+1) forecasting — 3D multivariate input', y=1.01)\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ── Cell 10: Final summary table ───────────────────────────────────
        nbf.v4.new_markdown_cell("## Final Summary"),
        nbf.v4.new_code_cell(
            "print('=' * 70)\n"
            "print(f\"{'Configuration':<30} | {'MG':>8} | {'NARMA':>8} | {'Lorenz':>8}\")\n"
            "print('-' * 70)\n\n"
            "rows = [\n"
            "    ('Qiskit default',           qk_mg[0][1], nm[0][1],  lz[0][1]),\n"
            "    ('Qiskit pers.+collapse',     qk_mg[2][1], None,      None),\n"
            "    ('Qiskit pers.+partial',      qk_mg[3][1], None,      None),\n"
            "    ('Perceval field',            pcv_mg[0][1], nm[4][1], None),\n"
            "    ('Perceval Fock analytic',    pcv_mg[1][1], nm[5][1], None),\n"
            "    ('Perceval Fock threshold',   pcv_mg[2][1], nm[6][1], None),\n"
            "    ('Perceval Fock feedback',    pcv_mg[3][1], nm[7][1], None),\n"
            "    ('Dynamiqs mixture',          dq_mg[0][1], nm[1][1],  lz[1][1]),\n"
            "    ('Dynamiqs Lindblad',         dq_mg[1][1], nm[2][1],  None),\n"
            "    ('Dynamiqs mixture+collapse', dq_mg[2][1], nm[3][1],  None),\n"
            "]\n\n"
            "for name, mg, narma, lorenz in rows:\n"
            "    mg_s    = f'{mg:.4f}'     if mg     is not None else '---'\n"
            "    nm_s    = f'{narma:.4f}'  if narma  is not None else '---'\n"
            "    lz_s    = f'{lorenz:.4f}' if lorenz is not None else '---'\n"
            "    print(f'{name:<30} | {mg_s:>8} | {nm_s:>8} | {lz_s:>8}')\n\n"
            "print('=' * 70)"
        ),
    ]

    notebook.cells = cells
    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote notebook to {notebook_path}")


if __name__ == "__main__":
    main()
