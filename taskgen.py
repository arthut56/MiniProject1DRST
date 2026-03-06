"""
Task Set Generator for DRTS Mini-project 1
============================================

Implements the UUniFast algorithm (Bini & Buttazzo, 2005) for generating
random task sets with a target total utilization.

Supports two deadline modes:
  - "implicit":    D_i = T_i  (RM and DM are equivalent)
  - "constrained": D_i ∈ [T_i/2, T_i] uniformly  (DM differs from RM)

Period ranges follow the approach from the RM vs EDF paper:
  T_i ∈ [T_min, T_max] uniformly distributed.
"""

from __future__ import annotations
import random
import math
from typing import List, Optional

from task import Task, TaskSet


# ═══════════════════════════════════════════════════════════════════════════
# UUniFast algorithm
# ═══════════════════════════════════════════════════════════════════════════

def uunifast(n: int, u_total: float, rng: random.Random) -> List[float]:
    """
    UUniFast algorithm for generating n utilizations summing to u_total.

    Guarantees a uniform distribution over the space of valid utilization
    vectors, unlike naive approaches.

    Reference: Bini & Buttazzo, "Measuring the Performance of
    Schedulability Tests", Real-Time Systems 30(1-2), 2005.
    """
    utilizations = []
    sum_u = u_total
    for i in range(1, n):
        # next_sum is uniformly distributed in [0, sum_u^{1/(n-i)}]
        next_sum = sum_u * (rng.random() ** (1.0 / (n - i)))
        utilizations.append(sum_u - next_sum)
        sum_u = next_sum
    utilizations.append(sum_u)
    return utilizations


def uunifast_discard(n: int, u_total: float, rng: random.Random,
                     max_attempts: int = 1000) -> Optional[List[float]]:
    """
    UUniFast-Discard: rejects any set where a single U_i > 1
    (which would require C_i > T_i, physically impossible).
    """
    for _ in range(max_attempts):
        utils = uunifast(n, u_total, rng)
        if all(0 < u <= 1.0 for u in utils):
            return utils
    return None  # Could not generate valid set


# ═══════════════════════════════════════════════════════════════════════════
# Task Set Generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_taskset(
    n_tasks: int = 10,
    utilization: float = 0.8,
    period_min: int = 10,
    period_max: int = 200,
    deadline_mode: str = "constrained",
    bcet_ratio: float = 0.1,
    seed: Optional[int] = None,
) -> Optional[TaskSet]:
    """
    Generate a random task set.

    Parameters
    ----------
    n_tasks : int
        Number of tasks.
    utilization : float
        Target total utilization (0 < U ≤ 1.0).
    period_min, period_max : int
        Range for task periods (uniformly distributed).
    deadline_mode : str
        "implicit"    → D_i = T_i
        "constrained" → D_i ∈ [C_i, T_i] (uniform, biased toward T_i/2..T_i)
    bcet_ratio : float
        BCET = max(1, round(bcet_ratio * WCET)).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    TaskSet or None if generation failed.
    """
    rng = random.Random(seed)

    # Generate per-task utilizations
    utils = uunifast_discard(n_tasks, utilization, rng)
    if utils is None:
        return None

    tasks = []
    for i, u_i in enumerate(utils):
        # Random period
        period = rng.randint(period_min, period_max)

        # WCET derived from utilization
        wcet = max(1, round(u_i * period))

        # Ensure C_i ≤ T_i
        wcet = min(wcet, period)

        # Deadline
        if deadline_mode == "implicit":
            deadline = period
        elif deadline_mode == "constrained":
            # D_i uniformly in [max(C_i, T_i/2), T_i]
            d_min = max(wcet, period // 2)
            deadline = rng.randint(d_min, period)
        else:
            raise ValueError(f"Unknown deadline_mode: {deadline_mode}")

        # BCET
        bcet = max(1, round(bcet_ratio * wcet))

        tasks.append(Task(
            task_id=i,
            bcet=bcet,
            wcet=wcet,
            period=period,
            deadline=deadline,
        ))

    return TaskSet(tasks=tasks)


def generate_tasksets(
    count: int,
    n_tasks: int = 10,
    utilization: float = 0.8,
    period_min: int = 10,
    period_max: int = 200,
    deadline_mode: str = "constrained",
    bcet_ratio: float = 0.1,
    base_seed: int = 0,
) -> List[TaskSet]:
    """Generate multiple task sets with different seeds."""
    tasksets = []
    for i in range(count):
        ts = generate_taskset(
            n_tasks=n_tasks,
            utilization=utilization,
            period_min=period_min,
            period_max=period_max,
            deadline_mode=deadline_mode,
            bcet_ratio=bcet_ratio,
            seed=base_seed + i,
        )
        if ts is not None:
            tasksets.append(ts)
    return tasksets


# ═══════════════════════════════════════════════════════════════════════════
# CSV export (compatible with course format)
# ═══════════════════════════════════════════════════════════════════════════

def export_taskset_csv(ts: TaskSet, filepath: str) -> None:
    """Export task set to course CSV format."""
    import csv
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TaskID", "Jitter", "BCET", "WCET",
                         "Period", "Deadline", "PE"])
        for t in ts.tasks:
            writer.writerow([t.task_id, t.jitter, t.bcet, t.wcet,
                             t.period, t.deadline, 0])


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== UUniFast test: 10 tasks, U=0.8, constrained deadlines ===")
    ts = generate_taskset(n_tasks=10, utilization=0.8,
                          deadline_mode="constrained", seed=42)
    if ts:
        ts.assign_dm_priorities()
        print(ts.summary())
        print(f"\nActual U = {ts.total_utilization:.4f}")
        has_constrained = any(t.deadline < t.period for t in ts.tasks)
        print(f"Has D_i < T_i tasks: {has_constrained}")

    print("\n=== Implicit deadlines (D=T) for comparison ===")
    ts2 = generate_taskset(n_tasks=10, utilization=0.8,
                           deadline_mode="implicit", seed=42)
    if ts2:
        ts2.assign_dm_priorities()
        print(ts2.summary())

    print("\n=== Batch: 5 sets at U=0.9 ===")
    sets = generate_tasksets(5, n_tasks=8, utilization=0.9, base_seed=100)
    for i, s in enumerate(sets):
        print(f"  Set {i}: n={s.n}, U={s.total_utilization:.4f}")
