import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from scheduler_analysis import (
    compute_hyperperiod,
    simulate_schedule,
)


REPORT_PLOT_TASKSETS = {
    "fig8": "task_sets/generated/report_fig8_taskset.csv",
    0.7: "task_sets/generated/report_fig9_u07_taskset.csv",
    0.8: "task_sets/generated/report_fig9_u08_taskset.csv",
    0.9: "task_sets/generated/report_fig9_u09_taskset.csv",
}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    tasks = tasks.copy()
    if "Task" in tasks.columns and "Name" not in tasks.columns:
        tasks = tasks.rename(columns={"Task": "Name"})
    required = ["Name", "BCET", "WCET", "Period", "Deadline"]
    missing = [c for c in required if c not in tasks.columns]
    if missing:
        raise ValueError(f"Missing required task columns: {missing}")
    return tasks[required]


def _load_generated_plot_taskset(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return _normalize_task_columns(pd.read_csv(path))


def _require_plot_taskset(path: str) -> pd.DataFrame:
    tasks = _load_generated_plot_taskset(path)
    if tasks is None:
        raise FileNotFoundError(
            f"Required plot taskset not found: {path}."
        )
    return tasks


def _save_or_return(fig, output_file=None):
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def _build_arj_taskset(target_utilization: float) -> pd.DataFrame:
    key = round(float(target_utilization), 1)
    if key not in REPORT_PLOT_TASKSETS:
        raise ValueError(f"No configured generated taskset for utilization {target_utilization}")
    return _require_plot_taskset(REPORT_PLOT_TASKSETS[key])
def plot_fraction_schedulable(sweep_df, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sweep_df['utilization'], sweep_df['dm_fraction_schedulable'],
           marker='o', linewidth=2.5, markersize=8, label='DM', color='#e74c3c')
    ax.plot(sweep_df['utilization'], sweep_df['edf_fraction_schedulable'],
           marker='s', linewidth=2.5, markersize=8, label='EDF', color='#2ecc71')
    ax.axvline(x=np.log(2), color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Processor Utilization U', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fraction Schedulable', fontsize=12, fontweight='bold')
    ax.set_title('Schedulability Comparison: DM vs EDF', fontsize=13, fontweight='bold')
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, output_file)
def plot_wcrt_comparison(results_df, output_file=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    valid = results_df[(pd.to_numeric(results_df['Ri_DM'], errors='coerce').notna()) &
                       (pd.to_numeric(results_df['Ri_EDF'], errors='coerce').notna())]
    if len(valid) == 0:
        return fig
    x = np.arange(len(valid))
    width = 0.35
    dm_wcrt = pd.to_numeric(valid['Ri_DM'])
    edf_wcrt = pd.to_numeric(valid['Ri_EDF'])
    ax.bar(x - width/2, dm_wcrt, width, label='DM', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, edf_wcrt, width, label='EDF', color='#2ecc71', alpha=0.8)
    ax.set_xlabel('Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('WCRT (time units)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Task WCRT Comparison: DM vs EDF', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"tau{i+1}" for i in range(len(valid))])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    return _save_or_return(fig, output_file)
def plot_analytical_vs_observed(results_df, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    dm_ana = pd.to_numeric(results_df['Ri_DM'], errors='coerce')
    dm_obs = pd.to_numeric(results_df['Ri_DM_obs_max'], errors='coerce')
    edf_ana = pd.to_numeric(results_df['Ri_EDF'], errors='coerce')
    edf_obs = pd.to_numeric(results_df['Ri_EDF_obs_max'], errors='coerce')
    valid_dm = ~(dm_ana.isna() | dm_obs.isna())
    valid_edf = ~(edf_ana.isna() | edf_obs.isna())
    ax.scatter(dm_ana[valid_dm], dm_obs[valid_dm], s=80, alpha=0.6, 
              label='DM', color='#e74c3c', edgecolors='black', linewidth=0.5)
    ax.scatter(edf_ana[valid_edf], edf_obs[valid_edf], s=80, alpha=0.6,
              label='EDF', color='#2ecc71', edgecolors='black', linewidth=0.5)
    max_val = max(dm_ana.max(), edf_ana.max(), dm_obs.max(), edf_obs.max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Analytical WCRT', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Max RT', fontsize=12, fontweight='bold')
    ax.set_title('Analytical vs Observed Response Times', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    return _save_or_return(fig, output_file)
def plot_preemptions(results_df, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if {'DM_preemptions', 'EDF_preemptions'}.issubset(results_df.columns):
        dm_col, edf_col = 'DM_preemptions', 'EDF_preemptions'
        title = 'Preemption Analysis: DM vs EDF (analysis task set)'
    elif {'dm_preemptions', 'edf_preemptions'}.issubset(results_df.columns):
        dm_col, edf_col = 'dm_preemptions', 'edf_preemptions'
        title = 'Preemption Analysis: DM vs EDF (all task sets)'
    else:
        raise ValueError('Preemption plot requires DM/EDF preemption columns')

    dm_preempt = pd.to_numeric(results_df[dm_col], errors='coerce').sum()
    edf_preempt = pd.to_numeric(results_df[edf_col], errors='coerce').sum()
    policies = ['DM', 'EDF']
    preempts = [dm_preempt, edf_preempt]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(policies, preempts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, preempts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(val)}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Preemptions', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(preempts) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    return _save_or_return(fig, output_file)


def _tc5_boxplot_taskset() -> pd.DataFrame:
    return _require_plot_taskset(REPORT_PLOT_TASKSETS["fig8"])


def _collect_stochastic_response_times(tasks: pd.DataFrame, policy: str,
                                       num_runs: int = 24,
                                       num_hyperperiods: int = 4,
                                       seed: int = 42) -> dict:
    """Collect per-task response-time samples from repeated stochastic simulations."""
    H = compute_hyperperiod(tasks["Period"].tolist())
    sim_time = H * max(1, int(num_hyperperiods))
    samples = {i: [] for i in range(len(tasks))}

    for run in range(num_runs):
        result = simulate_schedule(
            tasks,
            policy=policy,
            use_wcet=False,
            max_sim_time=sim_time,
            seed=seed + run,
        )
        for task_id, rts in result["response_times"].items():
            samples[task_id].extend(rts)

    return samples


def _build_fig8_samples_df() -> pd.DataFrame:
    """Build long-format samples for Figure 8 and persist as CSV upstream."""
    tasks = _tc5_boxplot_taskset()
    dm_samples = _collect_stochastic_response_times(tasks, "DM")
    edf_samples = _collect_stochastic_response_times(tasks, "EDF")

    rows = []
    for task_id in range(len(tasks)):
        task_name = tasks.loc[task_id, "Name"]
        for rt in dm_samples[task_id]:
            rows.append({"task_id": task_id, "task_name": task_name, "policy": "DM", "response_time": float(rt)})
        for rt in edf_samples[task_id]:
            rows.append({"task_id": task_id, "task_name": task_name, "policy": "EDF", "response_time": float(rt)})
    return pd.DataFrame(rows)


def _build_fig9_arj_df(utilizations=(0.7, 0.8, 0.9)) -> pd.DataFrame:
    """Build ARJ_i rows for Figure 9 and persist as CSV upstream."""
    rows = []
    for U in utilizations:
        tasks = _build_arj_taskset(U)
        dm_arj = _arj_per_task(tasks, "DM")
        edf_arj = _arj_per_task(tasks, "EDF")
        for task_id in range(len(tasks)):
            task_name = tasks.loc[task_id, "Name"]
            rows.append({"utilization": float(U), "task_id": task_id, "task_name": task_name, "policy": "DM", "arj": float(dm_arj[task_id])})
            rows.append({"utilization": float(U), "task_id": task_id, "task_name": task_name, "policy": "EDF", "arj": float(edf_arj[task_id])})
    return pd.DataFrame(rows)


def write_stochastic_plot_csvs(fig8_csv='data/fig8_tc5_rt_samples.csv',
                               fig9_csv='data/fig9_arj_u07_u08_u09.csv'):
    """Generate CSV data sources for Figures 8 and 9."""
    os.makedirs(os.path.dirname(fig8_csv) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(fig9_csv) or '.', exist_ok=True)
    _build_fig8_samples_df().to_csv(fig8_csv, index=False)
    _build_fig9_arj_df().to_csv(fig9_csv, index=False)


def _validate_fig8_data(fig8_df: pd.DataFrame, min_nonzero_deltas: int = 2):
    _require_columns(fig8_df, {"task_id", "task_name", "policy", "response_time"}, "Figure 8 CSV")
    deltas = 0
    for task_id in sorted(fig8_df["task_id"].dropna().unique().tolist()):
        dm = fig8_df[(fig8_df["task_id"] == task_id) & (fig8_df["policy"] == "DM")]["response_time"]
        edf = fig8_df[(fig8_df["task_id"] == task_id) & (fig8_df["policy"] == "EDF")]["response_time"]
        if dm.empty or edf.empty:
            continue
        if abs(float(dm.mean()) - float(edf.mean())) > 0.15 or abs(float(dm.max() - dm.min()) - float(edf.max() - edf.min())) > 0.5:
            deltas += 1
    if deltas < min_nonzero_deltas:
        raise ValueError(f"Figure 8 data not informative enough (nontrivial DM/EDF tasks={deltas})")


def _validate_fig9_data(fig9_df: pd.DataFrame, min_nonzero_pairs: int = 3):
    _require_columns(fig9_df, {"utilization", "task_id", "task_name", "policy", "arj"}, "Figure 9 CSV")
    pivot = fig9_df.pivot_table(index=["utilization", "task_id"], columns="policy", values="arj", aggfunc="first").dropna()
    if "DM" not in pivot.columns or "EDF" not in pivot.columns:
        raise ValueError("Figure 9 data missing DM/EDF ARJ values")
    nonzero = int((np.abs(pivot["DM"] - pivot["EDF"]) > 1e-9).sum())
    if nonzero < min_nonzero_pairs:
        raise ValueError(f"Figure 9 data not informative enough (nonzero DM/EDF pairs={nonzero})")


def plot_response_time_boxplots_tc5(fig8_df: pd.DataFrame, output_file=None):
    """Figure 8: Response-time distributions for TC5 under DM and EDF from CSV data."""
    _require_columns(fig8_df, {"task_id", "task_name", "policy", "response_time"}, "Figure 8 CSV")

    tasks_df = fig8_df[["task_id", "task_name"]].drop_duplicates().sort_values("task_id")
    task_names = tasks_df["task_name"].tolist()
    task_ids = tasks_df["task_id"].tolist()

    dm_df = fig8_df[fig8_df["policy"] == "DM"]
    edf_df = fig8_df[fig8_df["policy"] == "EDF"]

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(task_names)
    centers = np.arange(n)
    width = 0.32

    dm_data = [dm_df.loc[dm_df["task_id"] == task_id, "response_time"].tolist() or [0.0] for task_id in task_ids]
    edf_data = [edf_df.loc[edf_df["task_id"] == task_id, "response_time"].tolist() or [0.0] for task_id in task_ids]

    dm_pos = centers - width / 2
    edf_pos = centers + width / 2

    ax.boxplot(
        dm_data,
        positions=dm_pos,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#e74c3c", "alpha": 0.5},
        medianprops={"color": "black", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
    )
    ax.boxplot(
        edf_data,
        positions=edf_pos,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#2ecc71", "alpha": 0.5},
        medianprops={"color": "black", "linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.25},
    )

    ax.set_xticks(centers)
    ax.set_xticklabels(task_names)
    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Response Time (time units)", fontsize=12, fontweight="bold")
    ax.set_title("TC5 (Constrained-Deadline) Response-Time Distributions: DM vs EDF", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    legend_handles = [
        Patch(facecolor="#e74c3c", edgecolor="black", alpha=0.5, label="DM"),
        Patch(facecolor="#2ecc71", edgecolor="black", alpha=0.5, label="EDF"),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    return _save_or_return(fig, output_file)


def _arj_per_task(tasks: pd.DataFrame, policy: str) -> np.ndarray:
    """Compute ARJ_i = max(R_i,k) - min(R_i,k) for each task from stochastic samples."""
    samples = _collect_stochastic_response_times(tasks, policy)
    arj = []
    for task_id in range(len(tasks)):
        rts = samples[task_id]
        arj.append(float(max(rts) - min(rts)) if rts else 0.0)
    return np.array(arj)


def plot_arj_per_task_utilizations(fig9_df: pd.DataFrame, output_file=None):
    """Figure 9: ARJ per task under DM and EDF at U=0.7, 0.8, 0.9 from CSV data."""
    _require_columns(fig9_df, {"utilization", "task_id", "task_name", "policy", "arj"}, "Figure 9 CSV")

    utilizations = sorted(fig9_df["utilization"].dropna().unique().tolist())
    if not utilizations:
        raise ValueError("Figure 9 CSV has no utilization rows")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    if len(utilizations) != 3:
        raise ValueError(f"Figure 9 expects exactly 3 utilization levels, got: {utilizations}")

    for idx, U in enumerate(utilizations):
        u_df = fig9_df[np.isclose(fig9_df["utilization"], U)].copy()
        tasks_df = u_df[["task_id", "task_name"]].drop_duplicates().sort_values("task_id")
        task_ids = tasks_df["task_id"].tolist()
        dm_arj = [
            float(u_df[(u_df["task_id"] == task_id) & (u_df["policy"] == "DM")]["arj"].iloc[0])
            for task_id in task_ids
        ]
        edf_arj = [
            float(u_df[(u_df["task_id"] == task_id) & (u_df["policy"] == "EDF")]["arj"].iloc[0])
            for task_id in task_ids
        ]

        x = np.arange(len(task_ids))
        width = 0.36
        ax = axes[idx]
        ax.bar(x - width / 2, dm_arj, width, color="#e74c3c", alpha=0.8, label="DM")
        ax.bar(x + width / 2, edf_arj, width, color="#2ecc71", alpha=0.8, label="EDF")
        ax.set_ylabel("ARJ", fontsize=11, fontweight="bold")
        ax.set_title(f"ARJ per Task at U={U:.1f}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left")

    ordered_tasks = fig9_df[["task_id", "task_name"]].drop_duplicates().sort_values("task_id")
    axes[-1].set_xticks(np.arange(len(ordered_tasks)))
    axes[-1].set_xticklabels(ordered_tasks["task_name"].tolist())
    axes[-1].set_xlabel("Task", fontsize=12, fontweight="bold")
    fig.tight_layout()

    return _save_or_return(fig, output_file)
def generate_all_plots(analysis_csv='data/analysis_results.csv',
                      sweep_csv='data/fraction_schedulable_summary.csv',
                      fig8_csv='data/fig8_tc5_rt_samples.csv',
                      fig9_csv='data/fig9_arj_u07_u08_u09.csv',
                      all_tasksets_csv='data/all_tasksets_results.csv',
                      output_dir='data/figures'):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(analysis_csv):
        raise FileNotFoundError(f"Analysis CSV not found: {analysis_csv}")
    analysis_df = pd.read_csv(analysis_csv)
    plot_wcrt_comparison(analysis_df, os.path.join(output_dir, 'fig5_wcrt_comparison.png'))
    plot_analytical_vs_observed(analysis_df, os.path.join(output_dir, 'fig6_analytical_vs_observed.png'))
    preempt_df = pd.read_csv(all_tasksets_csv) if os.path.exists(all_tasksets_csv) else analysis_df
    plot_preemptions(preempt_df, os.path.join(output_dir, 'fig7_preemptions.png'))
    if sweep_csv and os.path.exists(sweep_csv):
        sweep_df = pd.read_csv(sweep_csv)
        plot_fraction_schedulable(sweep_df, os.path.join(output_dir, 'fig4_fraction_schedulable.png'))


    if not os.path.exists(fig8_csv):
        raise FileNotFoundError(f"Figure 8 CSV not found: {fig8_csv}")
    if not os.path.exists(fig9_csv):
        raise FileNotFoundError(f"Figure 9 CSV not found: {fig9_csv}")

    fig8_df = pd.read_csv(fig8_csv)
    fig9_df = pd.read_csv(fig9_csv)
    _validate_fig8_data(fig8_df)
    _validate_fig9_data(fig9_df)

    fig8_path = os.path.join(output_dir, 'fig8_tc5_rt_boxplot.png')
    fig9_path = os.path.join(output_dir, 'fig9_arj_u07_u08_u09.png')
    plot_response_time_boxplots_tc5(fig8_df, fig8_path)
    plot_arj_per_task_utilizations(fig9_df, fig9_path)

if __name__ == "__main__":
    generate_all_plots()
    print("Check under data/figures for plots")
