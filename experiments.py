"""Utilization sweep over pre-generated task sets."""

import os
import pandas as pd
from typing import List, Dict, Any
import random

from scheduler_analysis import (
    compute_utilization,
    dm_schedulability_test,
    edf_dbf_feasibility_test,
)
import uunifast

def _u_to_key(u: float) -> str:
    return f"u{int(round(float(u) * 100)):03d}"

def _load_sweep_tasksets(tasksets_dir: str, utilization: float) -> List[str]:
    util_dir = os.path.join(tasksets_dir, _u_to_key(utilization))
    if not os.path.isdir(util_dir):
        return []
    return sorted(
        os.path.join(util_dir, name)
        for name in os.listdir(util_dir)
        if name.endswith('.csv')
    )

def _normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    tasks = tasks.copy()
    if 'Task' in tasks.columns and 'Name' not in tasks.columns:
        tasks = tasks.rename(columns={'Task': 'Name'})
    required = ['Name', 'WCET', 'BCET', 'Period', 'Deadline']
    missing = [c for c in required if c not in tasks.columns]
    if missing:
        raise ValueError(f"Taskset missing required columns: {missing}")
    return tasks[required]


def test_taskset_verdicts(taskset: pd.DataFrame) -> Dict[str, Any]:
    """Test a single task set and return DM/EDF verdicts."""
    dm_ok, _, _ = dm_schedulability_test(taskset)
    edf_ok, _ = edf_dbf_feasibility_test(taskset)
    return {
        'dm_schedulable': dm_ok,
        'edf_feasible': edf_ok,
    }

def run_utilization_sweep(
    utilization_levels: List[float] = None,
    samples_per_level: int = 500,
    n_tasks: int = 10,
    tasksets_dir: str = 'task_sets/generated/sweep',
    output_dir: str = 'data',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run the required utilization sweep: load tasksets from directory if they exist.
    If not, generate them, save to CSV, and then use them.
    
    Args:
        utilization_levels: List of U values to test (default: [0.5, 0.6, ..., 1.0])
        samples_per_level: Number of random task sets per U level
        n_tasks: Number of tasks per generated task set (if None, randomized 10, 15, 20)
        tasksets_dir: Directory to save/load the CSV task sets
        output_dir: Directory for results CSV
        
    Returns:
        DataFrame with all results
    """
    if utilization_levels is None:
        utilization_levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    log = print if verbose else (lambda *args, **kwargs: None)
    log("=" * 80)
    log("UTILIZATION SWEEP EXPERIMENT (CACHED GENERATION)")
    log("=" * 80)
    log(f"Utilization levels: {utilization_levels}")
    log(f"Samples per level: {samples_per_level}")
    log(f"Tasks per set: Varied (10, 15, 20)" if n_tasks is None else f"Tasks per set: {n_tasks}")
    log(f"Tasksets directory: {tasksets_dir}")
    log()
    
    results = []
    sizes = [10, 15, 20]
    
    for U in utilization_levels:
        log(f"\nUtilization U = {U:.2f}:")
        log("-" * 60)
        
        util_dir = os.path.join(tasksets_dir, _u_to_key(U))
        os.makedirs(util_dir, exist_ok=True)
        
        existing_files = _load_sweep_tasksets(tasksets_dir, U)
        
        if len(existing_files) < samples_per_level:
            missing = samples_per_level - len(existing_files)
            log(f"  Generating {missing} missing task sets for U={U:.2f}...")
            start_idx = len(existing_files)
            for i in range(start_idx, samples_per_level):
                task_count = random.choice(sizes) if n_tasks is None else n_tasks
                taskset = uunifast.generate_constrained_taskset(task_count, U)
                out_file = os.path.join(util_dir, f"taskset_{i:04d}.csv")
                taskset.to_csv(out_file, index=False)
                
            existing_files = _load_sweep_tasksets(tasksets_dir, U)

        selected = existing_files[:samples_per_level]
        
        dm_sched_count = 0
        edf_sched_count = 0

        for sample_id, path in enumerate(selected):
            taskset = _normalize_task_columns(pd.read_csv(path))

            verdict = test_taskset_verdicts(taskset)
            dm_ok = verdict['dm_schedulable']
            edf_ok = verdict['edf_feasible']

            if dm_ok:
                dm_sched_count += 1
            if edf_ok:
                edf_sched_count += 1

            actual_u = compute_utilization(taskset)
            results.append({
                'utilization_target': U,
                'utilization_actual': actual_u,
                'n_tasks': len(taskset),
                'dm_schedulable': dm_ok,
                'edf_feasible': edf_ok,
                'sample_id': sample_id,
                'taskset_file': os.path.basename(path),
            })
            
            if (sample_id + 1) % 100 == 0:
                log(f"  Completed {sample_id + 1}/{samples_per_level} samples")
        
        dm_frac = dm_sched_count / samples_per_level
        edf_frac = edf_sched_count / samples_per_level
        log(f"  Results (n={samples_per_level}):")
        log(f"    DM schedulable: {dm_sched_count}/{samples_per_level} ({dm_frac:.1%})")
        log(f"    EDF schedulable: {edf_sched_count}/{samples_per_level} ({edf_frac:.1%})")
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'utilization_sweep_results.csv')
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    log("\n" + "=" * 80)
    log(f"Results saved to: {output_file}")
    log("=" * 80)
    
    return results_df


def compute_fraction_schedulable(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by utilization level to compute fraction schedulable.
    
    Returns:
        DataFrame with columns [utilization, dm_fraction, edf_fraction, n_samples]
    """
    summary = []
    
    for U in sorted(results_df['utilization_target'].unique()):
        subset = results_df[results_df['utilization_target'] == U]
        valid = subset[subset['dm_schedulable'].notna()]
        
        if len(valid) > 0:
            dm_frac = valid['dm_schedulable'].sum() / len(valid)
            edf_frac = valid['edf_feasible'].sum() / len(valid)
            
            summary.append({
                'utilization': U,
                'dm_fraction_schedulable': dm_frac,
                'edf_fraction_schedulable': edf_frac,
                'n_samples': len(valid)
            })
    
    return pd.DataFrame(summary)


if __name__ == "__main__":
    results_df = run_utilization_sweep(
        utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        samples_per_level=150,
        n_tasks=10,
        tasksets_dir='task_sets/generated/sweep',
        output_dir='data',
        verbose=True,
    )
    
    summary_df = compute_fraction_schedulable(results_df)
    summary_df.to_csv('data/fraction_schedulable_summary.csv', index=False)
    
    print("\nFraction Schedulable Summary:")
    print(summary_df.to_string(index=False))
