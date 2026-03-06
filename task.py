"""
Task model and CSV utilities for DRTS Mini-project 1.

Each periodic task τ_i is characterised by:
    - C_i  (WCET)   worst-case execution time
    - B_i  (BCET)   best-case execution time
    - D_i  (Deadline) relative deadline
    - T_i  (Period)
    - Constraint: C_i <= D_i <= T_i  (constrained deadlines)

Under Deadline Monotonic (DM), priority is inversely proportional to D_i
(shorter deadline ⟹ higher priority).
"""

from __future__ import annotations
import csv
import math
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Optional


@dataclass
class Task:
    """A single periodic real-time task."""
    task_id: int
    bcet: int          # Best-case execution time
    wcet: int          # Worst-case execution time  (C_i)
    period: int        # T_i
    deadline: int       # D_i  (relative deadline)
    jitter: int = 0    # Release jitter (ignored in basic analysis)
    priority: Optional[int] = None  # Assigned later by scheduling policy

    @property
    def utilization(self) -> float:
        """Processor utilization factor U_i = C_i / T_i."""
        return self.wcet / self.period

    def __repr__(self) -> str:
        return (f"Task(id={self.task_id}, C={self.wcet}, "
                f"D={self.deadline}, T={self.period}, U={self.utilization:.4f})")


@dataclass
class TaskSet:
    """An ordered collection of periodic tasks with convenience methods."""
    tasks: List[Task] = field(default_factory=list)

    # ── aggregate metrics ──────────────────────────────────────────────

    @property
    def total_utilization(self) -> float:
        """Total processor utilization U = Σ C_i / T_i."""
        return sum(t.utilization for t in self.tasks)

    @property
    def n(self) -> int:
        return len(self.tasks)

    @property
    def hyperperiod(self) -> int:
        """H = lcm(T_1, …, T_n). The schedule repeats every H time units."""
        return reduce(lambda a, b: a * b // math.gcd(a, b),
                      (t.period for t in self.tasks))

    # ── ordering helpers ───────────────────────────────────────────────

    def sorted_by_deadline(self) -> List[Task]:
        """Return tasks sorted by increasing relative deadline (DM order)."""
        return sorted(self.tasks, key=lambda t: t.deadline)

    def sorted_by_period(self) -> List[Task]:
        """Return tasks sorted by increasing period (RM order)."""
        return sorted(self.tasks, key=lambda t: t.period)

    def assign_dm_priorities(self) -> None:
        """Assign DM fixed priorities: smaller D_i ⟹ higher priority (lower number)."""
        for rank, t in enumerate(self.sorted_by_deadline()):
            t.priority = rank  # 0 = highest priority

    def assign_rm_priorities(self) -> None:
        """Assign RM fixed priorities: smaller T_i ⟹ higher priority."""
        for rank, t in enumerate(self.sorted_by_period()):
            t.priority = rank

    # ── display ────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"TaskSet  (n={self.n}, U={self.total_utilization:.4f}, "
                 f"H={self.hyperperiod})"]
        lines.append(f"{'ID':>4} {'BCET':>6} {'WCET':>6} {'Period':>8} "
                     f"{'Deadline':>8} {'U_i':>8} {'Prio':>5}")
        lines.append("-" * 54)
        for t in self.sorted_by_deadline():
            prio = t.priority if t.priority is not None else "-"
            lines.append(f"{t.task_id:>4} {t.bcet:>6} {t.wcet:>6} "
                         f"{t.period:>8} {t.deadline:>8} "
                         f"{t.utilization:>8.4f} {str(prio):>5}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TaskSet(n={self.n}, U={self.total_utilization:.4f})"


# ── CSV I/O ────────────────────────────────────────────────────────────────

def load_taskset_from_csv(filepath: str) -> TaskSet:
    """
    Load a task set from CSV. Supports two formats:

    Format 1 (course default):
        TaskID, Jitter, BCET, WCET, Period, Deadline, PE

    Format 2 (priority generator):
        Task, BCET, WCET, Period, Deadline, Priority

    PE and Priority columns are ignored for analysis.
    """
    tasks = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers is None:
            raise ValueError(f"Empty or invalid CSV: {filepath}")

        # Detect format based on column names
        has_taskid = "TaskID" in headers
        has_task = "Task" in headers
        has_jitter = "Jitter" in headers

        for row in reader:
            # Task ID
            if has_taskid:
                task_id = int(row["TaskID"])
            elif has_task:
                raw = row["Task"].strip()
                # Handle "Task_0", "Task_1" format
                if raw.startswith("Task_"):
                    task_id = int(raw.split("_")[1])
                else:
                    task_id = int(raw)
            else:
                raise ValueError(
                    f"CSV must have 'TaskID' or 'Task' column. "
                    f"Found: {headers}")

            # Jitter (optional, defaults to 0)
            jitter = int(row["Jitter"]) if has_jitter else 0

            # BCET (optional, defaults to WCET if missing)
            if "BCET" in headers:
                bcet = int(row["BCET"])
            else:
                bcet = int(row["WCET"])

            tasks.append(Task(
                task_id=task_id,
                jitter=jitter,
                bcet=bcet,
                wcet=int(row["WCET"]),
                period=int(row["Period"]),
                deadline=int(row["Deadline"]),
            ))
    return TaskSet(tasks=tasks)


def make_taskset(params: list[tuple[int, int, int]]) -> TaskSet:
    """
    Quick helper to build a TaskSet from a list of (C_i, D_i, T_i) tuples.
    BCET defaults to C_i (worst-case everywhere).
    """
    tasks = []
    for i, (c, d, t) in enumerate(params):
        tasks.append(Task(task_id=i, bcet=c, wcet=c, period=t, deadline=d))
    return TaskSet(tasks=tasks)
