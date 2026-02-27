"""
Test Schedulability of All Task Sets
=====================================
Runs schedulability analysis on all task sets in task_sets/ directory
and generates a summary report.
"""

import os
import pandas as pd
from scheduler_analysis import (
    compute_utilization,
    compute_hyperperiod,
    dm_schedulability_test,
    edf_dbf_feasibility_test,
    edf_wcrt_schedule_construction,
    simulate_schedule
)


def normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different CSV formats."""
    tasks = tasks.copy()
    if 'Task' in tasks.columns and 'Name' not in tasks.columns:
        tasks = tasks.rename(columns={'Task': 'Name'})
    return tasks


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
        'edf_preemptions': sum(edf_sim['preemptions'].values())
    }


def test_all_tasksets(base_dir: str = 'task_sets') -> pd.DataFrame:
    """
    Test all task sets in the given directory structure.

    Args:
        base_dir: Base directory containing schedulable/ and unschedulable/ folders

    Returns:
        DataFrame with all test results
    """
    results = []

    # Find all CSV files
    for category in ['schedulable', 'unschedulable']:
        category_path = os.path.join(base_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} not found")
            continue

        csv_files = [f for f in os.listdir(category_path) if f.endswith('.csv')]

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(category_path, csv_file)
            print(f"\nTesting: {category}/{csv_file}")
            print("-" * 60)

            try:
                result = test_task_set(csv_path)
                result['category'] = category
                result['expected_schedulable'] = (category == 'schedulable')

                # Check if result matches expectation
                result['dm_correct'] = (result['dm_schedulable'] == result['expected_schedulable'])
                result['edf_correct'] = (result['edf_feasible'] == result['expected_schedulable'])

                results.append(result)

                # Print summary
                print(f"   Tasks: {result['num_tasks']}, U: {result['utilization']:.4f}")
                print(f"   DM: {'✓' if result['dm_schedulable'] else '✗'} (expected: {'✓' if result['expected_schedulable'] else '✗'})")
                print(f"   EDF: {'✓' if result['edf_feasible'] else '✗'} (expected: {'✓' if result['expected_schedulable'] else '✗'})")
                print(f"   DM sim misses: {result['dm_sim_misses']}, EDF sim misses: {result['edf_sim_misses']}")

            except Exception as e:
                print(f"   ERROR: {str(e)}")
                results.append({
                    'file': csv_file,
                    'category': category,
                    'error': str(e)
                })

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print a summary of all test results."""
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL TASK SET TESTS")
    print("=" * 80)

    # Filter out errors
    valid_results = results_df[~results_df.get('error', pd.Series([None]*len(results_df))).notna()]

    if len(valid_results) == 0:
        print("No valid results to summarize.")
        return

    # Count by category
    for category in ['schedulable', 'unschedulable']:
        cat_results = valid_results[valid_results['category'] == category]
        if len(cat_results) == 0:
            continue

        print(f"\n{category.upper()} Task Sets ({len(cat_results)} files):")
        print("-" * 60)

        dm_correct = cat_results['dm_correct'].sum() if 'dm_correct' in cat_results else 0
        edf_correct = cat_results['edf_correct'].sum() if 'edf_correct' in cat_results else 0

        print(f"   DM correct predictions: {dm_correct}/{len(cat_results)}")
        print(f"   EDF correct predictions: {edf_correct}/{len(cat_results)}")

        # Show utilization range
        print(f"   Utilization range: {cat_results['utilization'].min():.4f} - {cat_results['utilization'].max():.4f}")

    # Overall comparison table
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

    # Run tests
    results_df = test_all_tasksets('task_sets')

    # Print summary
    print_summary(results_df)

    # Save results to CSV
    output_file = 'data/all_tasksets_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

