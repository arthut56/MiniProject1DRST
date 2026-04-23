"""
Test Schedulability of All Task Sets
"""

import os
import pandas as pd
from scheduler_analysis import (
    compute_utilization,
    compute_hyperperiod,
    dm_schedulability_test,
    edf_dbf_feasibility_test,
    simulate_schedule,
    normalize_task_columns as normalize_taskset,
)

# This file is an executable analysis script, not a pytest test module.
__test__ = False


def normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different CSV formats."""
    # Keep this wrapper for backwards compatibility with earlier imports.
    return normalize_taskset(tasks)


def test_task_set(csv_path: str) -> dict:
    """
    Test schedulability of a single task set.

    Returns:
        Dictionary with test results
    """
    tasks = pd.read_csv(csv_path)
    tasks = normalize_task_columns(tasks)

    n = len(tasks)
    U = compute_utilization(tasks)
    H = compute_hyperperiod(tasks["Period"].tolist())

    # DM Analysis
    dm_ok, dm_msg, tasks_dm = dm_schedulability_test(tasks)

    # EDF Analysis
    edf_ok, edf_msg = edf_dbf_feasibility_test(tasks)

    # Simulation (only if hyperperiod is reasonable)
    max_sim_time = min(H, 100000)

    dm_sim = simulate_schedule(tasks, policy="DM", use_wcet=True, max_sim_time=max_sim_time)
    edf_sim = simulate_schedule(tasks, policy="EDF", use_wcet=True, max_sim_time=max_sim_time)

    dm_misses = sum(dm_sim['deadline_misses'].values())
    edf_misses = sum(edf_sim['deadline_misses'].values())

    if (not dm_ok) and edf_ok:
        policy_classification = 'dm_only_unschedulable'
    elif (not dm_ok) and (not edf_ok):
        policy_classification = 'unschedulable_both'
    elif dm_ok and edf_ok:
        policy_classification = 'schedulable_both'
    else:
        policy_classification = 'dm_only_schedulable'

    return {
        'file': os.path.basename(csv_path),
        'num_tasks': n,
        'utilization': U,
        'hyperperiod': H,
        'dm_schedulable': dm_ok,
        'dm_message': dm_msg,
        'edf_feasible': edf_ok,
        'edf_message': edf_msg,
        'dm_sim_misses': dm_misses,
        'edf_sim_misses': edf_misses,
        'dm_preemptions': sum(dm_sim['preemptions'].values()),
        'edf_preemptions': sum(edf_sim['preemptions'].values()),
        'policy_classification': policy_classification,
    }


def test_all_tasksets(base_dir: str = 'task_sets', verbose: bool = False) -> pd.DataFrame:
    """
    Test all task sets in the given directory structure.

    Args:
        base_dir: Base directory containing schedulable/ and unschedulable/ folders

    Returns:
        DataFrame with all test results
    """
    results = []
    log = print if verbose else (lambda *args, **kwargs: None)

    for category in ['schedulable', 'unschedulable']:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            log(f"Warning: {category_path} not found")
            continue

        csv_files = [f for f in os.listdir(category_path) if f.endswith('.csv')]

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(category_path, csv_file)
            log(f"\nTesting: {category}/{csv_file}")
            log("-" * 60)

            try:
                result = test_task_set(csv_path)
                result['category'] = category
                result['dm_matches_wcet_sim'] = (
                    (result['dm_schedulable'] and result['dm_sim_misses'] == 0) or
                    ((not result['dm_schedulable']) and result['dm_sim_misses'] > 0)
                )
                result['edf_matches_wcet_sim'] = (
                    (result['edf_feasible'] and result['edf_sim_misses'] == 0) or
                    ((not result['edf_feasible']) and result['edf_sim_misses'] > 0)
                )

                results.append(result)

                log(f"   Tasks: {result['num_tasks']}, U: {result['utilization']:.4f}")
                log(f"   DM analytical: {result['dm_schedulable']}")
                log(f"   EDF analytical: {result['edf_feasible']}")
                log(f"   DM sim misses: {result['dm_sim_misses']}, EDF sim misses: {result['edf_sim_misses']}")

            except (OSError, ValueError, KeyError, pd.errors.ParserError) as e:
                log(f"   ERROR: {str(e)}")
                results.append({
                    'file': csv_file,
                    'category': category,
                    'error': str(e)
                })

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame, verbose: bool = True):
    """Print a summary of all test results."""
    if not verbose:
        return
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL TASK SET TESTS")
    print("=" * 80)

    valid_results = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()]

    if len(valid_results) == 0:
        print("No valid results to summarize.")
        return

    for category in ['schedulable', 'unschedulable']:
        cat_results = valid_results[valid_results['category'] == category]
        if len(cat_results) == 0:
            continue

        print(f"\n{category.upper()} Task Sets ({len(cat_results)} files):")
        print("-" * 60)

        dm_sched = cat_results['dm_schedulable'].sum() if 'dm_schedulable' in cat_results else 0
        edf_sched = cat_results['edf_feasible'].sum() if 'edf_feasible' in cat_results else 0
        dm_consistent = cat_results['dm_matches_wcet_sim'].sum() if 'dm_matches_wcet_sim' in cat_results else 0
        edf_consistent = cat_results['edf_matches_wcet_sim'].sum() if 'edf_matches_wcet_sim' in cat_results else 0

        print(f"   DM analytically schedulable: {dm_sched}/{len(cat_results)}")
        print(f"   EDF analytically schedulable: {edf_sched}/{len(cat_results)}")
        print(f"   DM verdict consistent with WCET simulation: {dm_consistent}/{len(cat_results)}")
        print(f"   EDF verdict consistent with WCET simulation: {edf_consistent}/{len(cat_results)}")

        if 'policy_classification' in cat_results:
            dm_only_unsched = (cat_results['policy_classification'] == 'dm_only_unschedulable').sum()
            both_unsched = (cat_results['policy_classification'] == 'unschedulable_both').sum()
            print(f"   DM-only unschedulable: {dm_only_unsched}/{len(cat_results)}")
            print(f"   Unschedulable under both DM and EDF: {both_unsched}/{len(cat_results)}")

        print(f"   Utilization range: {cat_results['utilization'].min():.4f} - {cat_results['utilization'].max():.4f}")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)

    cols_to_show = ['file', 'category', 'num_tasks', 'utilization',
                    'dm_schedulable', 'edf_feasible', 'dm_sim_misses', 'edf_sim_misses']
    cols_available = [c for c in cols_to_show if c in valid_results.columns]

    print(valid_results[cols_available].to_string(index=False))


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING ALL TASK SETS")
    print("=" * 80)

    results_df = test_all_tasksets('task_sets', verbose=True)
    print_summary(results_df, verbose=True)

    output_file = 'data/all_tasksets_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

