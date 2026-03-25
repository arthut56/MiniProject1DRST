"""
Spec Experiments: Utilization Sweep and Comparison Workflow
============================================================

Implements the required comparison methodology for utilization-based DM/EDF comparison:
- Generate task sets with varying utilization U ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}
- For each U, generate N=500 random task sets using UUniFast
- Run DM RTA and EDF PDC on each
- Record schedulability verdicts
- Plot: fraction schedulable vs utilization for DM and EDF
"""

import os
import random
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import csv

from scheduler_analysis import (
    compute_utilization,
    compute_hyperperiod,
    dm_schedulability_test,
    edf_dbf_feasibility_test,
)


def uunifast_generator(n: int, U: float, seed: int = None) -> List[float]:
    """
    Generate utilizations for n tasks using UUniFast algorithm.
    
    Returns list of utilizations that sum to approximately U.
    Reference: Bini & Buttazzo (2005)
    """
    if seed is not None:
        random.seed(seed)
    
    utilizations = []
    remaining = U
    
    for i in range(n - 1):
        # uniform distribution for next task
        next_util = remaining * random.random() ** (1.0 / (n - i))
        utilizations.append(next_util)
        remaining -= next_util
    
    utilizations.append(remaining)
    return utilizations


def generate_taskset(n_tasks: int, U: float, seed: int = None) -> pd.DataFrame:
    """
    Generate a random periodic task set with given utilization.
    
    Uses UUniFast for utilization distribution, then assigns random periods
    and derives WCET from utilization.
    
    Args:
        n_tasks: Number of tasks
        U: Total utilization target
        seed: Random seed
        
    Returns:
        DataFrame with columns [Name, WCET, BCET, Period, Deadline]
    """
    if seed is not None:
        random.seed(seed)
    
    utilizations = uunifast_generator(n_tasks, U, seed)
    
    # generate random periods in range [10, 1000]
    periods = [random.randint(10, 1000) for _ in range(n_tasks)]
    
    # compute WCET from utilization: C_i = U_i * T_i
    tasks_data = []
    for i in range(n_tasks):
        name = f"Task_{i}"
        period = periods[i]
        wcet = max(1, int(utilizations[i] * period))
        bcet = max(1, random.randint(1, wcet))
        deadline = period  # implicit deadline (D_i = T_i)
        
        tasks_data.append({
            'Name': name,
            'WCET': wcet,
            'BCET': bcet,
            'Period': period,
            'Deadline': deadline
        })
    
    df = pd.DataFrame(tasks_data)
    
    # verify/adjust utilization to be close to target
    actual_u = compute_utilization(df)
    if actual_u > 1.01:  # allow small overshoot
        # scale down WCETs proportionally
        scale_factor = U / actual_u
        df['WCET'] = (df['WCET'] * scale_factor).astype(int)
        df['WCET'] = df['WCET'].clip(lower=1)
    
    return df


def test_taskset_verdicts(taskset: pd.DataFrame) -> Dict[str, Any]:
    """Test a single task set and return DM/EDF verdicts."""
    try:
        # dm analysis
        dm_ok, _, _ = dm_schedulability_test(taskset)
        
        # edf analysis
        edf_ok, _ = edf_dbf_feasibility_test(taskset)
        
        return {
            'dm_schedulable': dm_ok,
            'edf_feasible': edf_ok,
            'error': None
        }
    except Exception as e:
        return {
            'dm_schedulable': None,
            'edf_feasible': None,
            'error': str(e)
        }


def run_utilization_sweep(
    utilization_levels: List[float] = None,
    samples_per_level: int = 500,
    n_tasks: int = 10,
    output_dir: str = 'data',
    seed: int = 42
) -> pd.DataFrame:
    """
    Run the required utilization sweep: generate task sets at each U level
    and record DM/EDF schedulability verdicts.
    
    Args:
        utilization_levels: List of U values to test (default: [0.5, 0.6, ..., 1.0])
        samples_per_level: Number of random task sets per U level
        n_tasks: Number of tasks per generated task set
        output_dir: Directory for results CSV
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with all results
    """
    if utilization_levels is None:
        utilization_levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    print("=" * 80)
    print("UTILIZATION SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Utilization levels: {utilization_levels}")
    print(f"Samples per level: {samples_per_level}")
    print(f"Tasks per set: {n_tasks}")
    print()
    
    results = []
    total_samples = len(utilization_levels) * samples_per_level
    sample_count = 0
    
    for U in utilization_levels:
        print(f"\nUtilization U = {U:.2f}:")
        print("-" * 60)
        
        dm_sched_count = 0
        edf_sched_count = 0
        error_count = 0
        
        for sample_id in range(samples_per_level):
            sample_count += 1
            
            # generate random task set
            taskset = generate_taskset(
                n_tasks=n_tasks,
                U=U,
                seed=seed + sample_count
            )
            
            # test verdicts
            verdict = test_taskset_verdicts(taskset)
            
            if verdict['error'] is None:
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
                    'n_tasks': n_tasks,
                    'dm_schedulable': dm_ok,
                    'edf_feasible': edf_ok,
                    'sample_id': sample_id
                })
            else:
                error_count += 1
                results.append({
                    'utilization_target': U,
                    'utilization_actual': None,
                    'n_tasks': n_tasks,
                    'dm_schedulable': None,
                    'edf_feasible': None,
                    'sample_id': sample_id
                })
            
            if (sample_id + 1) % 100 == 0:
                print(f"  Completed {sample_id + 1}/{samples_per_level} samples")
        
        # summary for this U level
        valid_samples = samples_per_level - error_count
        if valid_samples > 0:
            dm_frac = dm_sched_count / valid_samples
            edf_frac = edf_sched_count / valid_samples
            print(f"  Results (n={valid_samples}):")
            print(f"    DM schedulable: {dm_sched_count}/{valid_samples} ({dm_frac:.1%})")
            print(f"    EDF schedulable: {edf_sched_count}/{valid_samples} ({edf_frac:.1%})")
        
        if error_count > 0:
            print(f"  Errors: {error_count}")
    
    # save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'utilization_sweep_results.csv')
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
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
    # run the utilization sweep
    results_df = run_utilization_sweep(
        utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        samples_per_level=500,
        n_tasks=10,
        output_dir='data',
        seed=42
    )
    
    # compute summary statistics
    summary_df = compute_fraction_schedulable(results_df)
    summary_df.to_csv('data/fraction_schedulable_summary.csv', index=False)
    
    print("\nFraction Schedulable Summary:")
    print(summary_df.to_string(index=False))

