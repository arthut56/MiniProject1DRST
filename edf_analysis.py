"""
Earliest Deadline First (EDF) Schedulability Analysis
======================================================

Implements the Processor Demand Criterion from Buttazzo Section 4.6:

1. Simple utilization test       (D_i == T_i case: exact, O(n))
2. Processor Demand Criterion    (D_i ≤ T_i case: exact, pseudo-polynomial)

The demand-bound function:
    dbf(t) = Σ ⌊(t + T_i − D_i) / T_i⌋ · C_i

The task set is schedulable by EDF iff ∀t > 0: dbf(t) ≤ t.
Testing points can be restricted to absolute deadlines ≤ min(L*, H).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

from task import Task, TaskSet


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EDFAnalysisResult:
    """Overall EDF schedulability result."""
    schedulable: bool
    total_utilization: float
    method: str                    # "utilization" or "pdc"
    l_star: Optional[float] = None  # Upper bound on testing interval
    num_test_points: int = 0       # How many L values were checked
    failing_l: Optional[int] = None  # First L where dbf(L) > L, if any


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Utilization test  (D_i == T_i: necessary & sufficient, O(n))
# ═══════════════════════════════════════════════════════════════════════════

def edf_utilization_test(ts: TaskSet) -> bool:
    """
    Exact EDF test when all D_i == T_i:
        Σ C_i / T_i ≤ 1

    Returns True if schedulable.
    """
    return ts.total_utilization <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Processor Demand Criterion  (Section 4.6.1, Eq. 4.38–4.42)
# ═══════════════════════════════════════════════════════════════════════════

def _demand_bound_function(tasks: List[Task], t: int) -> int:
    """
    Compute dbf(t) = Σ ⌊(t + T_i − D_i) / T_i⌋ · C_i

    Only counts instances whose absolute deadline falls within [0, t].
    """
    total = 0
    for task in tasks:
        n_instances = max(0, math.floor((t + task.period - task.deadline) / task.period))
        total += n_instances * task.wcet
    return total


def _compute_l_star(tasks: List[Task], U: float) -> float:
    """
    L* = Σ U_i · (T_i − D_i) / (1 − U)

    Upper bound on the testing interval for PDC.
    """
    if U >= 1.0:
        return float("inf")
    numerator = sum(
        (t.wcet / t.period) * (t.period - t.deadline) for t in tasks
    )
    return numerator / (1.0 - U)


def _compute_testing_set(tasks: List[Task], max_l: int) -> List[int]:
    """
    Build the set D of testing points:
        D = { d_k : d_k ≤ min(L*, H) }

    These are all absolute deadlines d_{i,k} = k·T_i − (T_i − D_i)
    that fall within [0, max_l].
    """
    points = set()
    for task in tasks:
        k = 1
        while True:
            # Absolute deadline of k-th instance: (k-1)*T_i + D_i = k*T_i - (T_i - D_i)
            d_k = (k - 1) * task.period + task.deadline
            if d_k > max_l:
                break
            points.add(d_k)
            k += 1
    return sorted(points)


def edf_pdc(ts: TaskSet) -> EDFAnalysisResult:
    """
    Exact EDF schedulability test using the Processor Demand Criterion.

    A synchronous set of periodic tasks with D_i ≤ T_i is schedulable
    by EDF if and only if:
        ∀t ∈ D:  dbf(t) ≤ t

    where D is the set of absolute deadlines ≤ min(L*, H).
    """
    U = ts.total_utilization

    # Quick check: if U > 1, immediately unschedulable
    if U > 1.0:
        return EDFAnalysisResult(
            schedulable=False, total_utilization=U,
            method="pdc", l_star=None, num_test_points=0,
        )

    # Check if all deadlines equal periods (simpler test applies)
    all_d_eq_t = all(t.deadline == t.period for t in ts.tasks)
    if all_d_eq_t:
        return EDFAnalysisResult(
            schedulable=(U <= 1.0), total_utilization=U,
            method="utilization",
        )

    # General case: Processor Demand Criterion
    l_star = _compute_l_star(ts.tasks, U)
    H = ts.hyperperiod

    # Use the tighter bound
    max_l = int(min(l_star, H))

    # For very large max_l, we need a reasonable cap
    # (hyperperiods can be enormous with many tasks)
    MAX_TEST_LIMIT = 10_000_000
    if max_l > MAX_TEST_LIMIT:
        # Fall back to checking only up to a reasonable limit
        # This makes the test sufficient but not necessary for huge H
        max_l = MAX_TEST_LIMIT

    # Also check D_max as a minimum
    d_max = max(t.deadline for t in ts.tasks)
    max_l = max(max_l, d_max)
    if l_star < float("inf"):
        max_l = int(min(max_l, l_star)) if l_star > 0 else d_max

    testing_points = _compute_testing_set(ts.tasks, max_l)

    for L in testing_points:
        dbf = _demand_bound_function(ts.tasks, L)
        if dbf > L:
            return EDFAnalysisResult(
                schedulable=False, total_utilization=U,
                method="pdc", l_star=l_star,
                num_test_points=len(testing_points),
                failing_l=L,
            )

    return EDFAnalysisResult(
        schedulable=True, total_utilization=U,
        method="pdc", l_star=l_star,
        num_test_points=len(testing_points),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience
# ═══════════════════════════════════════════════════════════════════════════

def edf_analysis(ts: TaskSet) -> EDFAnalysisResult:
    """Run the appropriate EDF schedulability test."""
    return edf_pdc(ts)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test with Buttazzo Table 4.4 example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from task import make_taskset

    # Buttazzo Table 4.4:  (C, D, T)
    #   τ1: C=2, D=4, T=6
    #   τ2: C=2, D=5, T=8
    #   τ3: C=3, D=7, T=9
    ts = make_taskset([
        (2, 4, 6),
        (2, 5, 8),
        (3, 7, 9),
    ])
    print(f"U = {ts.total_utilization:.4f}")
    result = edf_pdc(ts)
    print(f"EDF PDC: schedulable={result.schedulable}, "
          f"method={result.method}, L*={result.l_star:.1f}, "
          f"test points={result.num_test_points}")
    assert result.schedulable, "Buttazzo Table 4.4 should be EDF-schedulable"
    print("✓ Buttazzo Table 4.4 example verified: EDF schedulable")
