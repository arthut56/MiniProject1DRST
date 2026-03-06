"""
Deadline Monotonic (DM) Schedulability Analysis
================================================

Implements three levels of analysis from Buttazzo Chapter 4, Section 4.5:

1. Sufficient utilization-based test   (quick, pessimistic)
2. Response-time upper bound           (sufficient, not necessary)
3. Response Time Analysis – RTA        (exact, necessary & sufficient)

RTA (Figure 4.17) is the primary tool: it computes the exact worst-case
response time (WCRT) for every task and checks R_i ≤ D_i.
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
class DMAnalysisResult:
    """Holds per-task WCRT results from DM analysis."""
    task_id: int
    wcet: int
    deadline: int
    period: int
    wcrt: Optional[int]        # None if unschedulable (did not converge within D_i)
    schedulable: bool
    iterations: int            # Number of RTA iterations to converge

    @property
    def slack(self) -> Optional[int]:
        """D_i − R_i: positive means margin, negative means deadline miss."""
        if self.wcrt is None:
            return None
        return self.deadline - self.wcrt


@dataclass
class DMResult:
    """Complete DM analysis result for a task set."""
    task_results: List[DMAnalysisResult]
    schedulable: bool
    total_utilization: float

    def summary(self) -> str:
        lines = [
            f"DM Schedulability: {'SCHEDULABLE' if self.schedulable else 'UNSCHEDULABLE'}",
            f"Total utilization: {self.total_utilization:.4f}",
            "",
            f"{'ID':>4} {'WCET':>6} {'D_i':>8} {'T_i':>8} "
            f"{'WCRT':>8} {'Slack':>8} {'Iters':>6} {'OK?':>5}",
            "-" * 58,
        ]
        for r in self.task_results:
            wcrt_str = str(r.wcrt) if r.wcrt is not None else "N/A"
            slack_str = str(r.slack) if r.slack is not None else "N/A"
            ok = "yes" if r.schedulable else "NO"
            lines.append(
                f"{r.task_id:>4} {r.wcet:>6} {r.deadline:>8} {r.period:>8} "
                f"{wcrt_str:>8} {slack_str:>8} {r.iterations:>6} {ok:>5}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Sufficient utilization-based test  (Section 4.5.1, pessimistic)
# ═══════════════════════════════════════════════════════════════════════════

def dm_utilization_test(ts: TaskSet) -> bool:
    """
    Sufficient (but pessimistic) DM schedulability test:
        Σ C_i / D_i  ≤  n · (2^{1/n} − 1)

    Returns True if the task set PASSES (guaranteed schedulable).
    A False does NOT prove unschedulability.
    """
    n = ts.n
    util_by_deadline = sum(t.wcet / t.deadline for t in ts.tasks)
    bound = n * (2 ** (1.0 / n) - 1)
    return util_by_deadline <= bound


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Response-time upper bound  (Section 4.5.2, Eq. 4.20–4.21)
# ═══════════════════════════════════════════════════════════════════════════

def dm_response_time_upper_bound(ts: TaskSet) -> DMResult:
    """
    Sufficient approximate test using the closed-form upper bound:

        R_i^{ub} = (Σ_{h=1..i} C_h) / (1 − Σ_{h=1..i−1} U_h)

    Faster than RTA but may reject schedulable task sets.
    """
    ordered = ts.sorted_by_deadline()
    results: List[DMAnalysisResult] = []
    all_ok = True

    sum_c = 0
    sum_u = 0.0

    for i, task in enumerate(ordered):
        sum_c += task.wcet
        denom = 1.0 - sum_u
        if denom <= 0:
            # Utilization of higher-priority tasks already ≥ 1
            results.append(DMAnalysisResult(
                task_id=task.task_id, wcet=task.wcet,
                deadline=task.deadline, period=task.period,
                wcrt=None, schedulable=False, iterations=0,
            ))
            all_ok = False
        else:
            r_ub = math.ceil(sum_c / denom)
            ok = r_ub <= task.deadline
            if not ok:
                all_ok = False
            results.append(DMAnalysisResult(
                task_id=task.task_id, wcet=task.wcet,
                deadline=task.deadline, period=task.period,
                wcrt=r_ub, schedulable=ok, iterations=0,
            ))
        sum_u += task.utilization

    return DMResult(
        task_results=results,
        schedulable=all_ok,
        total_utilization=ts.total_utilization,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Response Time Analysis – RTA  (Section 4.5.2, Figure 4.17)
#     Exact, necessary & sufficient
# ═══════════════════════════════════════════════════════════════════════════

def _compute_initial_guess(task: Task, idx: int,
                           higher_priority: List[Task],
                           prev_wcrt: Optional[int]) -> int:
    """
    Optimised initial guess combining Methods 2 & 3 from Section 4.5.3:

        R_i^{(0)} = max( R_{i−1} + C_i ,  C_i / (1 − Σ_{h<i} U_h) )

    Falls back to C_i when no previous data is available.
    """
    # Method 2: previous response time + own computation
    if prev_wcrt is not None:
        guess_m2 = prev_wcrt + task.wcet
    else:
        guess_m2 = task.wcet

    # Method 3: lower bound from removing ceiling
    sum_u_hp = sum(t.utilization for t in higher_priority)
    denom = 1.0 - sum_u_hp
    if denom > 0:
        guess_m3 = math.ceil(task.wcet / denom)
    else:
        guess_m3 = task.wcet

    return max(guess_m2, guess_m3)


def dm_rta(ts: TaskSet, use_optimised_guess: bool = True) -> DMResult:
    """
    Exact DM schedulability test via Response Time Analysis.

    For each task τ_i (ordered by increasing D_i):
        R_i^{(0)} = C_i  (or optimised guess)
        R_i^{(s)} = C_i + Σ_{h : D_h < D_i} ⌈R_i^{(s−1)} / T_h⌉ · C_h

    Converges when R_i^{(s)} == R_i^{(s−1)}.
    Stops early if R_i^{(s)} > D_i  ⟹  unschedulable.

    Parameters
    ----------
    ts : TaskSet
        The task set to analyse (priorities assigned internally by DM order).
    use_optimised_guess : bool
        If True, uses Section 4.5.3 Methods 2+3 for the initial guess.

    Returns
    -------
    DMResult with per-task WCRTs and overall schedulability verdict.
    """
    ordered = ts.sorted_by_deadline()
    results: List[DMAnalysisResult] = []
    all_ok = True
    prev_wcrt: Optional[int] = None

    for i, task in enumerate(ordered):
        higher_priority = ordered[:i]

        # ── initial guess ──────────────────────────────────────────────
        if use_optimised_guess and i > 0:
            r = _compute_initial_guess(task, i, higher_priority, prev_wcrt)
        else:
            r = task.wcet

        # ── iterative convergence ──────────────────────────────────────
        iterations = 0
        converged = False
        while True:
            iterations += 1
            interference = sum(
                math.ceil(r / hp.period) * hp.wcet
                for hp in higher_priority
            )
            r_new = task.wcet + interference

            if r_new == r:
                converged = True
                break
            if r_new > task.deadline:
                # Unschedulable – response time exceeds deadline
                converged = False
                break
            r = r_new

        if converged and r <= task.deadline:
            ok = True
            prev_wcrt = r
        else:
            ok = False
            r = r_new if not converged else r
            all_ok = False
            prev_wcrt = None  # Can't rely on this for next guess

        results.append(DMAnalysisResult(
            task_id=task.task_id, wcet=task.wcet,
            deadline=task.deadline, period=task.period,
            wcrt=r if ok else r,
            schedulable=ok, iterations=iterations,
        ))

    return DMResult(
        task_results=results,
        schedulable=all_ok,
        total_utilization=ts.total_utilization,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: run all DM tests at once
# ═══════════════════════════════════════════════════════════════════════════

def dm_full_analysis(ts: TaskSet) -> dict:
    """Run all three DM tests and return a dict of results."""
    ts.assign_dm_priorities()
    return {
        "utilization_test": dm_utilization_test(ts),
        "upper_bound": dm_response_time_upper_bound(ts),
        "rta": dm_rta(ts, use_optimised_guess=True),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Quick self-test with Buttazzo Table 4.3 example
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from task import make_taskset

    # Buttazzo Table 4.3:  (C, D, T)
    #   τ1: C=1, T=4,  D=3
    #   τ2: C=1, T=5,  D=4
    #   τ3: C=2, T=6,  D=5
    #   τ4: C=1, T=11, D=10
    ts = make_taskset([
        (1, 3, 4),
        (1, 4, 5),
        (2, 5, 6),
        (1, 10, 11),
    ])
    ts.assign_dm_priorities()
    print(ts.summary())
    print()

    result = dm_rta(ts)
    print(result.summary())
    print()

    # Expected: R4 = 10, all schedulable
    # Verify τ4 specifically:
    r4 = [r for r in result.task_results if r.task_id == 3][0]
    assert r4.wcrt == 10, f"Expected R4=10, got {r4.wcrt}"
    assert r4.schedulable, "τ4 should be schedulable"
    print("✓ Buttazzo Table 4.3 example verified: R4 = 10, all schedulable")
