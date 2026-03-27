#!/usr/bin/env python3
"""
Master Runner: Complete Experimental Workflow
==============================================

Executes the full spec-required pipeline:
1. Utilization sweep (U={0.5..1.0}, N=500 samples) with DM/EDF comparison
2. Main scheduler analysis on curated task sets
3. Batch validation on all provided test cases
4. Generate all required visualizations
5. Summary report

Usage:
    python3 run_all_experiments.py [--quick]
    
Options:
    --quick    Skip utilization sweep (for fast testing)
"""

import sys
import os
import argparse

def run_workflow(quick=False):
    """Execute the complete experimental workflow."""
    
    print("=" * 80)
    print("MINI-PROJECT 1: COMPLETE EXPERIMENTAL WORKFLOW")
    print("=" * 80)
    print()
    
    # step 1: main scheduler analysis
    print("STEP 1: Main Scheduler Analysis")
    print("-" * 80)
    import scheduler_analysis
    tasks = __import__('pandas').read_csv('task_sets/schedulable/Full_Utilization_NonUnique_Periods_taskset.csv')
    tasks = scheduler_analysis.normalize_task_columns(tasks)
    results = scheduler_analysis.analyze_task_set(tasks, num_sim_runs=10, num_hyperperiods=1, seed=42)
    os.makedirs('data', exist_ok=True)
    results['results_df'].to_csv('data/analysis_results.csv', index=False)
    print("Saved: data/analysis_results.csv")
    print()
    
    # step 2: batch test all task sets
    print("STEP 2: Batch Validation on All Task Sets")
    print("-" * 80)
    import test_all_tasksets
    results_df = test_all_tasksets.test_all_tasksets('task_sets')
    test_all_tasksets.print_summary(results_df)
    results_df.to_csv('data/all_tasksets_results.csv', index=False)
    print()
    
    # step 3: utilization sweep (optional)
    if not quick:
        print("STEP 3: Utilization Sweep (500 samples per U level)")
        print("-" * 80)
        import experiments
        sweep_results = experiments.run_utilization_sweep(
            utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            samples_per_level=500,
            n_tasks=10,
            output_dir='data',
            seed=42
        )
        summary = experiments.compute_fraction_schedulable(sweep_results)
        summary.to_csv('data/fraction_schedulable_summary.csv', index=False)
        print()
    else:
        print("STEP 3: Utilization Sweep - SKIPPED (--quick mode)")
        print()
    
    # step 4: generate gantt charts
    print("STEP 4: Generate Gantt Charts")
    print("-" * 80)
    import generate_gantt_charts
    generate_gantt_charts.main()
    print()

    # step 5: generate visualizations
    print("STEP 5: Generate Visualizations")
    print("-" * 80)
    figure_dir = 'data/figures'
    sweep_summary_path = 'data/fraction_schedulable_summary.csv'
    sweep_input = sweep_summary_path if os.path.exists(sweep_summary_path) else None
    import visualizations
    visualizations.generate_all_plots(
        analysis_csv='data/analysis_results.csv',
        sweep_csv=sweep_summary_path if not quick else sweep_input,
        output_dir=figure_dir
    )
    print()

    print(f"Figures can be found in {figure_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete experimental workflow runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python3 run_all_experiments.py --quick"
    )
    parser.add_argument('--quick', action='store_true',
                       help='Skip utilization sweep for faster testing')
    args = parser.parse_args()
    
    try:
        run_workflow(quick=args.quick)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

