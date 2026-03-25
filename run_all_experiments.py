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
from pathlib import Path

def run_workflow(quick=False):
    """Execute the complete experimental workflow."""
    
    print("=" * 80)
    print("MINI-PROJECT 1: COMPLETE EXPERIMENTAL WORKFLOW")
    print("=" * 80)
    print()
    
    # Step 1: Main scheduler analysis
    print("STEP 1: Main Scheduler Analysis")
    print("-" * 80)
    import scheduler_analysis
    tasks = __import__('pandas').read_csv('task_sets/schedulable/Full_Utilization_NonUnique_Periods_taskset.csv')
    tasks = scheduler_analysis.normalize_task_columns(tasks)
    results = scheduler_analysis.analyze_task_set(tasks, num_sim_runs=10, num_hyperperiods=1, seed=42)
    print()
    
    # Step 2: Batch test all task sets
    print("STEP 2: Batch Validation on All Task Sets")
    print("-" * 80)
    import test_all_tasksets
    results_df = test_all_tasksets.test_all_tasksets('task_sets')
    test_all_tasksets.print_summary(results_df)
    results_df.to_csv('data/all_tasksets_results.csv', index=False)
    print()
    
    # Step 3: Utilization sweep (optional)
    if not quick:
        print("STEP 3: Utilization Sweep (500 samples per U level)")
        print("-" * 80)
        import spec_experiments
        sweep_results = spec_experiments.run_utilization_sweep(
            utilization_levels=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            samples_per_level=500,
            n_tasks=10,
            output_dir='data',
            seed=42
        )
        summary = spec_experiments.compute_fraction_schedulable(sweep_results)
        summary.to_csv('data/fraction_schedulable_summary.csv', index=False)
        print()
    else:
        print("STEP 3: Utilization Sweep - SKIPPED (--quick mode)")
        print()
    
    # Step 4: Generate visualizations
    print("STEP 4: Generate Visualizations")
    print("-" * 80)
    import visualizations
    visualizations.generate_all_plots(
        analysis_csv='data/analysis_results.csv',
        sweep_csv='data/fraction_schedulable_summary.csv' if not quick else None,
        output_dir='report/figures'
    )
    print()
    
    # Step 5: Summary
    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print()
    print("Generated outputs:")
    print(f"  ✓ data/analysis_results.csv")
    print(f"  ✓ data/all_tasksets_results.csv")
    if not quick:
        print(f"  ✓ data/fraction_schedulable_summary.csv")
        print(f"  ✓ data/utilization_sweep_results.csv")
    print(f"  ✓ report/figures/fig*.png (multiple)")
    print()
    print("Next steps:")
    print("  1. Inspect figures in report/figures/")
    print("  2. Review CSV outputs in data/")
    print("  3. Build PDF report: cd report && pdflatex main.tex")
    print()


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

