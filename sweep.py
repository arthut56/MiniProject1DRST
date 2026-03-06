"""
Utilization Sweep: DM vs EDF Comparison
========================================

Generates many task sets across a range of utilizations and compares:
  1. Schedulability ratio: % of task sets schedulable by DM vs EDF
  2. WCRT comparison: analytical vs simulated for both algorithms
  3. Deadline miss rates from simulation

Produces publication-ready plots for the project report.

Usage:
    python sweep.py                    # Run with defaults
    python sweep.py --sets 200         # More samples per utilization
    python sweep.py --mode implicit    # D_i = T_i only
    python sweep.py --mode constrained # D_i < T_i (more interesting)
    python sweep.py --mode both        # Run both modes
"""

from __future__ import annotations
import argparse
import os
import sys
import time

from task import TaskSet
from taskgen import generate_taskset, generate_tasksets
from dm_analysis import dm_rta
from edf_analysis import edf_pdc


# ═══════════════════════════════════════════════════════════════════════════
# Core sweep logic
# ═══════════════════════════════════════════════════════════════════════════

def run_sweep(
    utilizations: list[float],
    sets_per_util: int = 100,
    n_tasks: int = 10,
    period_min: int = 10,
    period_max: int = 200,
    deadline_mode: str = "constrained",
) -> dict:
    """
    For each target utilization, generate many task sets and test
    schedulability under DM (RTA) and EDF (PDC).

    Returns dict with per-utilization results.
    """
    results = {
        "utilizations": utilizations,
        "dm_sched_ratio": [],
        "edf_sched_ratio": [],
        "both_sched": [],       # schedulable by both
        "edf_only": [],         # schedulable by EDF but not DM
        "neither": [],          # not schedulable by either
        "dm_util_test_ratio": [],  # DM utilization sufficient test
        "avg_dm_wcrt_ratio": [],   # avg (WCRT / D_i) for DM-schedulable sets
        "deadline_mode": deadline_mode,
        "n_tasks": n_tasks,
        "sets_per_util": sets_per_util,
    }

    total_steps = len(utilizations)
    for step, u_target in enumerate(utilizations):
        dm_ok = 0
        edf_ok = 0
        both_ok = 0
        edf_only_count = 0
        neither_count = 0
        dm_util_ok = 0
        wcrt_ratios = []

        tasksets = generate_tasksets(
            count=sets_per_util,
            n_tasks=n_tasks,
            utilization=u_target,
            period_min=period_min,
            period_max=period_max,
            deadline_mode=deadline_mode,
            base_seed=step * 10000,
        )

        for ts in tasksets:
            ts.assign_dm_priorities()

            # DM analysis (RTA)
            dm_result = dm_rta(ts)
            dm_schedulable = dm_result.schedulable

            # EDF analysis (PDC)
            edf_result = edf_pdc(ts)
            edf_schedulable = edf_result.schedulable

            if dm_schedulable:
                dm_ok += 1
            if edf_schedulable:
                edf_ok += 1
            if dm_schedulable and edf_schedulable:
                both_ok += 1
            if edf_schedulable and not dm_schedulable:
                edf_only_count += 1
            if not dm_schedulable and not edf_schedulable:
                neither_count += 1

            # DM utilization sufficient test
            from dm_analysis import dm_utilization_test
            if dm_utilization_test(ts):
                dm_util_ok += 1

            # Average WCRT/D ratio for schedulable tasks
            if dm_schedulable:
                ratios = [
                    r.wcrt / r.deadline
                    for r in dm_result.task_results
                    if r.schedulable and r.wcrt is not None
                ]
                if ratios:
                    wcrt_ratios.append(sum(ratios) / len(ratios))

        n = len(tasksets)
        results["dm_sched_ratio"].append(dm_ok / n if n > 0 else 0)
        results["edf_sched_ratio"].append(edf_ok / n if n > 0 else 0)
        results["both_sched"].append(both_ok / n if n > 0 else 0)
        results["edf_only"].append(edf_only_count / n if n > 0 else 0)
        results["neither"].append(neither_count / n if n > 0 else 0)
        results["dm_util_test_ratio"].append(dm_util_ok / n if n > 0 else 0)
        results["avg_dm_wcrt_ratio"].append(
            sum(wcrt_ratios) / len(wcrt_ratios) if wcrt_ratios else 0
        )

        # Progress
        pct = (step + 1) / total_steps * 100
        print(f"  [{pct:5.1f}%] U={u_target:.2f}: "
              f"DM={dm_ok}/{n} ({100*dm_ok/n:.0f}%), "
              f"EDF={edf_ok}/{n} ({100*edf_ok/n:.0f}%), "
              f"EDF-only={edf_only_count}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, output_dir: str = ".") -> list[str]:
    """Generate comparison plots. Returns list of saved file paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        print("Install with: pip install matplotlib")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved = []
    mode = results["deadline_mode"]
    n = results["n_tasks"]

    u = results["utilizations"]

    # ── Plot 1: Schedulability ratio ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(u, [x * 100 for x in results["dm_sched_ratio"]],
            "b-o", markersize=4, linewidth=2, label="DM (RTA exact)")
    ax.plot(u, [x * 100 for x in results["edf_sched_ratio"]],
            "r-s", markersize=4, linewidth=2, label="EDF (PDC exact)")
    ax.plot(u, [x * 100 for x in results["dm_util_test_ratio"]],
            "b--^", markersize=3, linewidth=1, alpha=0.6,
            label="DM (utilization sufficient test)")

    # Mark the RM/DM Ulub = n(2^{1/n} - 1)
    import math
    ulub = n * (2 ** (1.0 / n) - 1)
    ax.axvline(x=ulub, color="blue", linestyle=":", alpha=0.5)
    ax.text(ulub + 0.01, 50, f"DM U_lub={ulub:.2f}",
            color="blue", fontsize=9, rotation=90, va="center")

    ax.axvline(x=1.0, color="red", linestyle=":", alpha=0.5)
    ax.text(1.01, 50, "EDF bound=1.0",
            color="red", fontsize=9, rotation=90, va="center")

    ax.set_xlabel("Target Utilization", fontsize=12)
    ax.set_ylabel("Schedulable Task Sets (%)", fontsize=12)
    ax.set_title(f"DM vs EDF Schedulability  "
                 f"(n={n}, {results['sets_per_util']} sets/point, "
                 f"deadlines={mode})", fontsize=13)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xlim(u[0] - 0.02, u[-1] + 0.02)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"schedulability_{mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── Plot 2: EDF-only gap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(u, [x * 100 for x in results["both_sched"]],
                    color="green", alpha=0.4, label="Schedulable by both")
    ax.fill_between(u,
                    [x * 100 for x in results["both_sched"]],
                    [(x + y) * 100 for x, y in
                     zip(results["both_sched"], results["edf_only"])],
                    color="orange", alpha=0.5,
                    label="EDF-only (DM fails, EDF succeeds)")
    ax.fill_between(u,
                    [(x + y) * 100 for x, y in
                     zip(results["both_sched"], results["edf_only"])],
                    [100] * len(u),
                    color="red", alpha=0.2,
                    label="Neither schedulable")

    ax.set_xlabel("Target Utilization", fontsize=12)
    ax.set_ylabel("Task Sets (%)", fontsize=12)
    ax.set_title(f"DM vs EDF Gap  (n={n}, deadlines={mode})", fontsize=13)
    ax.legend(fontsize=11, loc="center left")
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"edf_gap_{mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    # ── Plot 3: Average WCRT/Deadline ratio ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    # Filter out zeros (no schedulable sets at high U)
    valid = [(ui, wr) for ui, wr in
             zip(u, results["avg_dm_wcrt_ratio"]) if wr > 0]
    if valid:
        vu, vw = zip(*valid)
        ax.plot(vu, vw, "b-o", markersize=4, linewidth=2)
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5,
                   label="WCRT = Deadline (limit)")
        ax.set_xlabel("Target Utilization", fontsize=12)
        ax.set_ylabel("Average WCRT / Deadline", fontsize=12)
        ax.set_title(f"DM: How Close to Deadline?  "
                     f"(n={n}, deadlines={mode})", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"wcrt_ratio_{mode}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    print(f"  Saved: {path}")

    return saved


def print_summary_table(results: dict) -> None:
    """Print a text summary table of the sweep results."""
    print(f"\n{'U':>6} {'DM%':>6} {'EDF%':>6} {'EDF-only%':>10} "
          f"{'DM-Util%':>9} {'WCRT/D':>8}")
    print("-" * 50)
    for i, u in enumerate(results["utilizations"]):
        print(f"{u:>6.2f} "
              f"{results['dm_sched_ratio'][i]*100:>5.1f}% "
              f"{results['edf_sched_ratio'][i]*100:>5.1f}% "
              f"{results['edf_only'][i]*100:>9.1f}% "
              f"{results['dm_util_test_ratio'][i]*100:>8.1f}% "
              f"{results['avg_dm_wcrt_ratio'][i]:>8.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DM vs EDF Utilization Sweep"
    )
    parser.add_argument("--sets", type=int, default=100,
                        help="Task sets per utilization point (default: 100)")
    parser.add_argument("--tasks", type=int, default=10,
                        help="Tasks per set (default: 10)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["implicit", "constrained", "both"],
                        help="Deadline mode (default: both)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots (default: results)")
    args = parser.parse_args()

    utilizations = [round(x * 0.05, 2) for x in range(5, 21)]
    # i.e. [0.25, 0.30, 0.35, ..., 0.95, 1.00]

    modes = ["implicit", "constrained"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'='*65}")
        print(f"  SWEEP: {mode} deadlines, n={args.tasks}, "
              f"{args.sets} sets/point")
        print(f"  Utilizations: {utilizations[0]:.2f} → {utilizations[-1]:.2f}")
        print(f"{'='*65}\n")

        t0 = time.time()
        results = run_sweep(
            utilizations=utilizations,
            sets_per_util=args.sets,
            n_tasks=args.tasks,
            deadline_mode=mode,
        )
        elapsed = time.time() - t0
        print(f"\n  Completed in {elapsed:.1f}s")

        print_summary_table(results)

        print(f"\nGenerating plots...")
        saved = plot_results(results, output_dir=args.output)
        if not saved:
            print("  (No plots — install matplotlib: pip install matplotlib)")

    print(f"\nDone! Check the '{args.output}/' directory for plots.")


if __name__ == "__main__":
    main()
