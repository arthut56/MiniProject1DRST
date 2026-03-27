"""Visualization module for real-time scheduling analysis."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scheduler_analysis import compute_hyperperiod, simulate_schedule


MAX_SIM_TIME_FOR_PLOTS = 25000


def _save_or_return(fig, output_file=None):
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close(fig)
    return fig


def _placeholder_figure(title: str, message: str, output_file=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return _save_or_return(fig, output_file)


def _build_arj_taskset(target_utilization: float) -> pd.DataFrame:
    """Build a 6-task set with non-harmonic periods optimised for ARJ visualisation.

    Uses BCET=1 for all tasks so that the stochastic execution range [1, WCET]
    is maximised, making the DM vs EDF jitter difference clearly visible:
    - tau_1 (highest DM priority): RT = execution_time only → small ARJ
    - tau_6 (lowest DM priority): RT includes variable interference from all
      higher-priority tasks → large ARJ under DM
    - EDF distributes delays more evenly across all tasks.
    """
    periods = np.array([10, 23, 50, 110, 250, 500], dtype=int)
    per_task_u = target_utilization / len(periods)
    wcets = np.maximum(1, np.floor(per_task_u * periods).astype(int))
    max_wcet = np.maximum(1, periods // 2)
    wcets = np.minimum(wcets, max_wcet)
    return pd.DataFrame({
        "Name": [f"tau_{i + 1}" for i in range(len(periods))],
        "WCET": wcets,
        "BCET": np.ones(len(periods), dtype=int),   # BCET=1: maximise jitter range
        "Period": periods,
        "Deadline": periods,
    })
def _spec_tc1() -> pd.DataFrame:
    """Spec TC1: U=0.708, both DM and EDF schedulable (§8.2)."""
    return pd.DataFrame([
        {"Name": "tau_1", "BCET": 1, "WCET": 1, "Period": 4, "Deadline": 4},
        {"Name": "tau_2", "BCET": 2, "WCET": 2, "Period": 6, "Deadline": 6},
        {"Name": "tau_3", "BCET": 1, "WCET": 1, "Period": 8, "Deadline": 8},
    ])


def plot_wcrt_tc1_tc5(output_file=None):
    """Figure 5: per-task WCRT comparison (DM vs EDF) for spec TC1 and TC5."""
    from scheduler_analysis import dm_rta, edf_wcrt_schedule_construction

    tc1 = _spec_tc1()
    tc5 = _tc5_taskset()

    tc1_dm, _ = dm_rta(tc1)
    tc1_edf, _ = edf_wcrt_schedule_construction(tc1)
    tc5_dm, _ = dm_rta(tc5)
    tc5_edf, _ = edf_wcrt_schedule_construction(tc5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dm_df, edf_df, title, n in [
        (axes[0], tc1_dm, tc1_edf, "TC1  (U=0.708, n=3)", len(tc1)),
        (axes[1], tc5_dm, tc5_edf, "TC5  (U=0.890, n=6)", len(tc5)),
    ]:
        dm_wcrt = pd.to_numeric(dm_df["Ri_DM"], errors="coerce").fillna(0).values
        edf_wcrt = pd.to_numeric(edf_df["Ri_EDF"], errors="coerce").fillna(0).values
        x = np.arange(n)
        w = 0.35
        ax.bar(x - w / 2, dm_wcrt, w, label="DM", color="#e74c3c", alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.bar(x + w / 2, edf_wcrt, w, label="EDF", color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"τ{i+1}" for i in range(n)])
        ax.set_xlabel("Task", fontsize=11, fontweight="bold")
        ax.set_ylabel("WCRT (time units)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Task WCRT: DM vs EDF", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_or_return(fig, output_file)


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
    dm_preempt = pd.to_numeric(results_df['DM_preemptions'], errors='coerce').sum()
    edf_preempt = pd.to_numeric(results_df['EDF_preemptions'], errors='coerce').sum()
    policies = ['DM', 'EDF']
    preempts = [dm_preempt, edf_preempt]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(policies, preempts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, preempts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(val)}', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Preemptions', fontsize=12, fontweight='bold')
    ax.set_title('Preemption Analysis: DM vs EDF', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(preempts) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    return _save_or_return(fig, output_file)


def _tc5_taskset() -> pd.DataFrame:
    """Return the TC5 task set from the report specification."""
    return pd.DataFrame([
        {"Name": "tau_1", "BCET": 1, "WCET": 2, "Period": 10, "Deadline": 10},
        {"Name": "tau_2", "BCET": 1, "WCET": 2, "Period": 20, "Deadline": 20},
        {"Name": "tau_3", "BCET": 2, "WCET": 6, "Period": 40, "Deadline": 40},
        {"Name": "tau_4", "BCET": 2, "WCET": 7, "Period": 50, "Deadline": 50},
        {"Name": "tau_5", "BCET": 3, "WCET": 12, "Period": 80, "Deadline": 80},
        {"Name": "tau_6", "BCET": 5, "WCET": 15, "Period": 100, "Deadline": 100},
    ])


def _collect_stochastic_response_times(tasks: pd.DataFrame, policy: str,
                                       num_runs: int = 24,
                                       num_hyperperiods: int = 4,
                                       seed: int = 42) -> dict:
    """Collect per-task response-time samples from repeated stochastic simulations."""
    H = compute_hyperperiod(tasks["Period"].tolist())
    sim_time = min(H * max(1, int(num_hyperperiods)), MAX_SIM_TIME_FOR_PLOTS)
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


def plot_response_time_boxplots_tc5(output_file=None):
    """Figure 8: Response-time distributions for TC5 under DM and EDF."""
    tasks = _tc5_taskset()
    dm_samples = _collect_stochastic_response_times(tasks, "DM")
    edf_samples = _collect_stochastic_response_times(tasks, "EDF")

    fig, ax = plt.subplots(figsize=(12, 6))
    n = len(tasks)
    centers = np.arange(n)
    width = 0.32

    dm_data = [dm_samples[i] if dm_samples[i] else [0] for i in range(n)]
    edf_data = [edf_samples[i] if edf_samples[i] else [0] for i in range(n)]

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
    ax.set_xticklabels(tasks["Name"].tolist())
    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Response Time (time units)", fontsize=12, fontweight="bold")
    ax.set_title("TC5 Response-Time Distributions: DM vs EDF", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(["DM", "EDF"], loc="upper left")

    return _save_or_return(fig, output_file)


def _arj_per_task(tasks: pd.DataFrame, policy: str) -> np.ndarray:
    """Compute ARJ_i = max(R_i,k) - min(R_i,k) for each task from stochastic samples."""
    samples = _collect_stochastic_response_times(tasks, policy)
    arj = []
    for task_id in range(len(tasks)):
        rts = samples[task_id]
        arj.append(float(max(rts) - min(rts)) if rts else 0.0)
    return np.array(arj)


def plot_arj_per_task_utilizations(output_file=None):
    """Figure 9: ARJ per task under DM and EDF at U=0.7, 0.8, 0.9.

    Uses non-harmonic periods with BCET=1 so the DM vs EDF difference is
    visible: tau_1 (highest DM priority) has low ARJ under DM because it
    never experiences interference; tau_6 (lowest priority) has large ARJ
    under DM due to variable interference from all five higher-priority tasks.
    EDF distributes delays more evenly across all tasks.
    """
    utilizations = [0.7, 0.8, 0.9]
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    for idx, U in enumerate(utilizations):
        tasks = _build_arj_taskset(U)

        dm_arj = _arj_per_task(tasks, "DM")
        edf_arj = _arj_per_task(tasks, "EDF")

        x = np.arange(len(tasks))
        width = 0.36
        ax = axes[idx]
        ax.bar(x - width / 2, dm_arj, width, color="#e74c3c", alpha=0.8, label="DM")
        ax.bar(x + width / 2, edf_arj, width, color="#2ecc71", alpha=0.8, label="EDF")
        ax.set_ylabel("ARJ", fontsize=11, fontweight="bold")
        ax.set_title(f"ARJ per Task at U={U:.1f}", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left")

    axes[-1].set_xticks(np.arange(6))
    axes[-1].set_xticklabels([f"tau_{i + 1}" for i in range(6)])
    axes[-1].set_xlabel("Task", fontsize=12, fontweight="bold")
    fig.tight_layout()

    return _save_or_return(fig, output_file)
def generate_all_plots(analysis_csv='data/analysis_results.csv',
                      sweep_csv='data/fraction_schedulable_summary.csv',
                      output_dir='data/figures'):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    # Fig5: WCRT comparison for spec TC1 and TC5 (always generated analytically)
    plot_wcrt_tc1_tc5(os.path.join(output_dir, 'fig5_wcrt_comparison.png'))

    if os.path.exists(analysis_csv):
        print(f"Loading: {analysis_csv}")
        analysis_df = pd.read_csv(analysis_csv)
        plot_analytical_vs_observed(analysis_df, os.path.join(output_dir, 'fig6_analytical_vs_observed.png'))
        plot_preemptions(analysis_df, os.path.join(output_dir, 'fig7_preemptions.png'))
    else:
        print(f"Warning: {analysis_csv} not found")
    if sweep_csv and os.path.exists(sweep_csv):
        print(f"Loading: {sweep_csv}")
        sweep_df = pd.read_csv(sweep_csv)
        plot_fraction_schedulable(sweep_df, os.path.join(output_dir, 'fig4_fraction_schedulable.png'))
    elif sweep_csv:
        print("Warning: sweep summary CSV not found; skipping fraction schedulable plot")

    # figures required by report spec with fallback placeholders if heavy steps fail
    fig8_path = os.path.join(output_dir, 'fig8_tc5_rt_boxplot.png')
    fig9_path = os.path.join(output_dir, 'fig9_arj_u07_u08_u09.png')

    try:
        plot_response_time_boxplots_tc5(fig8_path)
    except Exception as exc:
        _placeholder_figure(
            "Figure 8 Placeholder",
            f"response-time boxplot generation failed: {exc}",
            fig8_path,
        )

    try:
        plot_arj_per_task_utilizations(fig9_path)
    except Exception as exc:
        _placeholder_figure(
            "Figure 9 Placeholder",
            f"arj plot generation failed: {exc}",
            fig9_path,
        )

    print("=" * 70)
    print(f"Plots saved to: {output_dir}")
    print("=" * 70)
if __name__ == "__main__":
    generate_all_plots()
