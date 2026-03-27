#!/usr/bin/env python3
"""
Experimental workflow runner.
"""

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
    os.makedirs('data/figures', exist_ok=True)

    print("Running main scheduler analysis...")
    tasks = pd.read_csv('task_sets/schedulable/Full_Utilization_NonUnique_Periods_taskset.csv')
    tasks = scheduler_analysis.normalize_task_columns(tasks)
    results = scheduler_analysis.analyze_task_set(tasks, num_sim_runs=10, num_hyperperiods=1, seed=42, verbose=False)
    results['results_df'].to_csv('data/analysis_results.csv', index=False)
    
    print("Running batch validation on all task sets...")
    results_df = test_all_tasksets.test_all_tasksets('task_sets', verbose=False)
    results_df.to_csv('data/all_tasksets_results.csv', index=False)
    
    if not quick:
        print("Running utilization sweep... (This may take a minute or two)")
        sweep_results = experiments.run_utilization_sweep(
            utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            samples_per_level=400,
            n_tasks=None,
            tasksets_dir='task_sets/generated/sweep',
            output_dir='data',
            verbose=False,
        )
        summary = experiments.compute_fraction_schedulable(sweep_results)
        summary.to_csv('data/fraction_schedulable_summary.csv', index=False)
    else:
        print("Skipping utilization sweep (--quick mode)")
    
    print("Generating gantt charts...")
    generate_gantt_charts.main()


    print("Generating visualizations...")
    figure_dir = 'data/figures'
    sweep_input = 'data/fraction_schedulable_summary.csv' if not quick else None


    visualizations.write_stochastic_plot_csvs(
        fig8_csv='data/fig8_tc5_rt_samples.csv',
        fig9_csv='data/fig9_arj_u07_u08_u09.csv',
    )
    
    visualizations.generate_all_plots(
        analysis_csv='data/analysis_results.csv',
        sweep_csv=sweep_input,
        fig8_csv='data/fig8_tc5_rt_samples.csv',
        fig9_csv='data/fig9_arj_u07_u08_u09.csv',
        all_tasksets_csv='data/all_tasksets_results.csv',
        output_dir=figure_dir
    )

    print(f"Workflow complete. Figures saved in {figure_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental workflow runner")
    parser.add_argument('--quick', action='store_true', help='Skip utilization sweep')
    args = parser.parse_args()
    
    run_workflow(quick=args.quick)
