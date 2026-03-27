#!/usr/bin/env python3
"""
Experimental workflow runner.
"""

import sys
import os
import argparse
import pandas as pd

import scheduler_analysis
import test_all_tasksets
import experiments
import generate_gantt_charts
import visualizations

def run_workflow(quick=False):
    os.makedirs('data', exist_ok=True)
    
    print("Running main scheduler analysis...")
    tasks = pd.read_csv('task_sets/schedulable/Full_Utilization_NonUnique_Periods_taskset.csv')
    tasks = scheduler_analysis.normalize_task_columns(tasks)
    results = scheduler_analysis.analyze_task_set(tasks, num_sim_runs=10, num_hyperperiods=1, seed=42)
    results['results_df'].to_csv('data/analysis_results.csv', index=False)
    
    print("Running batch validation on all task sets...")
    results_df = test_all_tasksets.test_all_tasksets('task_sets')
    test_all_tasksets.print_summary(results_df)
    results_df.to_csv('data/all_tasksets_results.csv', index=False)
    
    if not quick:
        print("Running utilization sweep...")
        sweep_results = experiments.run_utilization_sweep(
            utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            samples_per_level=500,
            n_tasks=10,
            output_dir='data',
            seed=42
        )
        summary = experiments.compute_fraction_schedulable(sweep_results)
        summary.to_csv('data/fraction_schedulable_summary.csv', index=False)
    else:
        print("Skipping utilization sweep (--quick mode)")
    
    print("Generating gantt charts...")
    generate_gantt_charts.main()

    print("Generating visualizations...")
    figure_dir = 'data/figures'
    sweep_summary_path = 'data/fraction_schedulable_summary.csv'
    sweep_input = sweep_summary_path if os.path.exists(sweep_summary_path) and not quick else None
    
    visualizations.generate_all_plots(
        analysis_csv='data/analysis_results.csv',
        sweep_csv=sweep_input,
        output_dir=figure_dir
    )

    print(f"Workflow complete. Figures saved in {figure_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental workflow runner")
    parser.add_argument('--quick', action='store_true', help='Skip utilization sweep')
    args = parser.parse_args()
    
    try:
        run_workflow(quick=args.quick)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
