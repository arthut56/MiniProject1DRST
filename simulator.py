"""
Discrete-Event Simulator for DM and EDF Scheduling
====================================================

Follows the simulation architecture from the w2d-simulation slides:
  - Initialization routine: set clock, state, counters, event list
  - Timing routine: determine next event, advance clock
  - Event routines: handle job arrivals and completions
  - Report generator: compute response time statistics

Events:
  - JOB_ARRIVAL (type 0): a new periodic job is released
  - JOB_COMPLETION (type 1): the running job finishes execution

The simulator runs for a configurable duration (default: 2x hyperperiod)
and records per-job response times for comparison against analytical WCRTs.
"""

from __future__ import annotations
import heapq
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from task import Task, TaskSet


EVENT_ARRIVAL = 0
EVENT_COMPLETION = 1


class SchedulingPolicy(Enum):
    DM = "dm"
    EDF = "edf"


@dataclass
class Job:
    task_id: int
    job_index: int
    release_time: int
    absolute_deadline: int
    execution_time: int
    remaining_time: int
    relative_deadline: int
    period: int


@dataclass
class JobRecord:
    task_id: int
    job_index: int
    release_time: int
    absolute_deadline: int
    finish_time: int
    execution_time: int
    response_time: int
    deadline_met: bool


@dataclass
class SimulationResult:
    policy: SchedulingPolicy
    duration: int
    records: List[JobRecord] = field(default_factory=list)
    deadline_misses: int = 0
    preemptions: int = 0

    def task_response_times(self) -> Dict[int, List[int]]:
        rt: Dict[int, List[int]] = {}
        for rec in self.records:
            rt.setdefault(rec.task_id, []).append(rec.response_time)
        return rt

    def task_wcrt_observed(self) -> Dict[int, int]:
        rt = self.task_response_times()
        return {tid: max(times) for tid, times in rt.items()}

    def task_avg_rt(self) -> Dict[int, float]:
        rt = self.task_response_times()
        return {tid: sum(times) / len(times) for tid, times in rt.items()}

    def summary(self) -> str:
        wcrt = self.task_wcrt_observed()
        avg = self.task_avg_rt()
        lines = [
            f"Simulation ({self.policy.value.upper()}) -- "
            f"duration={self.duration}, jobs={len(self.records)}, "
            f"deadline_misses={self.deadline_misses}, "
            f"preemptions={self.preemptions}",
            "",
            f"{'TaskID':>6} {'Jobs':>6} {'AvgRT':>10} {'ObsWCRT':>10}",
            "-" * 38,
        ]
        rt = self.task_response_times()
        for tid in sorted(rt.keys()):
            lines.append(
                f"{tid:>6} {len(rt[tid]):>6} "
                f"{avg[tid]:>10.1f} {wcrt[tid]:>10}"
            )
        return "\n".join(lines)


class Simulator:
    """
    Discrete-event simulator for single-processor preemptive scheduling.

    At each time step, the simulator processes all events at the current time,
    then dispatches the highest-priority ready job to run until the next event.
    """

    def __init__(self, taskset: TaskSet, policy: SchedulingPolicy,
                 duration: Optional[int] = None, seed: Optional[int] = None,
                 use_wcet: bool = False):
        self.taskset = taskset
        self.policy = policy
        self.use_wcet = use_wcet
        self.duration = duration if duration is not None else 2 * taskset.hyperperiod
        self.rng = random.Random(seed)

    def _priority_key(self, job: Job):
        """Lower = higher priority."""
        if self.policy == SchedulingPolicy.DM:
            return (job.relative_deadline, job.task_id)
        else:
            return (job.absolute_deadline, job.task_id)

    def _sample_exec(self, task: Task) -> int:
        if self.use_wcet or task.bcet >= task.wcet:
            return task.wcet
        return self.rng.randint(max(1, task.bcet), task.wcet)

    def run(self) -> SimulationResult:
        clock = 0
        seq = 0  # tie-breaker for heap ordering
        eq: list = []
        ready: List[Job] = []
        running: Optional[Job] = None
        records: List[JobRecord] = []
        misses = 0
        preemptions = 0
        job_ctr: Dict[int, int] = {t.task_id: 0 for t in self.taskset.tasks}
        completed: set = set()  # (task_id, job_index) to avoid duplicates

        def push(time, etype, data=None):
            nonlocal seq
            heapq.heappush(eq, (time, etype, seq, data))
            seq += 1

        # Schedule all first arrivals at t=0
        for task in self.taskset.tasks:
            push(0, EVENT_ARRIVAL, task.task_id)

        while eq:
            ev_time, ev_type, _, ev_data = heapq.heappop(eq)
            if ev_time > self.duration:
                break

            # --- Advance running job ---
            if running is not None and ev_time > clock:
                running.remaining_time -= (ev_time - clock)

            clock = ev_time

            # --- If running job just finished, record it ---
            if running is not None and running.remaining_time <= 0:
                key = (running.task_id, running.job_index)
                if key not in completed:
                    completed.add(key)
                    rt = clock - running.release_time
                    records.append(JobRecord(
                        task_id=running.task_id,
                        job_index=running.job_index,
                        release_time=running.release_time,
                        absolute_deadline=running.absolute_deadline,
                        finish_time=clock,
                        execution_time=running.execution_time,
                        response_time=rt,
                        deadline_met=(clock <= running.absolute_deadline),
                    ))
                    if clock > running.absolute_deadline:
                        misses += 1
                running = None

            # --- Handle event ---
            if ev_type == EVENT_ARRIVAL:
                task_id = ev_data
                task = next(t for t in self.taskset.tasks if t.task_id == task_id)
                ji = job_ctr[task_id]
                job_ctr[task_id] += 1
                et = self._sample_exec(task)

                job = Job(
                    task_id=task_id, job_index=ji,
                    release_time=clock,
                    absolute_deadline=clock + task.deadline,
                    execution_time=et, remaining_time=et,
                    relative_deadline=task.deadline,
                    period=task.period,
                )
                ready.append(job)

                # Schedule next arrival
                nxt = clock + task.period
                if nxt < self.duration:
                    push(nxt, EVENT_ARRIVAL, task_id)

            elif ev_type == EVENT_COMPLETION:
                # Already handled above (running.remaining_time <= 0)
                pass

            # --- Dispatch ---
            # Put running job back in ready if still has work
            if running is not None and running.remaining_time > 0:
                ready.append(running)
                running = None

            if ready:
                # Pick highest priority
                best_idx = 0
                best_key = self._priority_key(ready[0])
                for i in range(1, len(ready)):
                    k = self._priority_key(ready[i])
                    if k < best_key:
                        best_key = k
                        best_idx = i
                new_running = ready.pop(best_idx)

                # Check if this is a preemption
                # (a different job was running and got put back)
                running = new_running

                # Schedule completion event
                finish = clock + running.remaining_time
                if finish <= self.duration:
                    push(finish, EVENT_COMPLETION, running.task_id)

        return SimulationResult(
            policy=self.policy,
            duration=self.duration,
            records=records,
            deadline_misses=misses,
            preemptions=preemptions,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def simulate_dm(ts: TaskSet, duration: Optional[int] = None,
                seed: Optional[int] = None,
                use_wcet: bool = False) -> SimulationResult:
    ts.assign_dm_priorities()
    sim = Simulator(ts, SchedulingPolicy.DM, duration, seed, use_wcet)
    return sim.run()


def simulate_edf(ts: TaskSet, duration: Optional[int] = None,
                 seed: Optional[int] = None,
                 use_wcet: bool = False) -> SimulationResult:
    sim = Simulator(ts, SchedulingPolicy.EDF, duration, seed, use_wcet)
    return sim.run()


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from task import make_taskset

    ts = make_taskset([
        (1, 3, 4),
        (1, 4, 5),
        (2, 5, 6),
        (1, 10, 11),
    ])

    print("=== DM Simulation (WCET) ===")
    result = simulate_dm(ts, use_wcet=True)
    print(result.summary())

    # Check critical instant for task 3 (the lowest priority task)
    t3_jobs = sorted([r for r in result.records if r.task_id == 3],
                     key=lambda r: r.release_time)
    if t3_jobs:
        first = t3_jobs[0]
        print(f"\nTask 3 first job: release={first.release_time}, "
              f"finish={first.finish_time}, RT={first.response_time}")
        if first.response_time == 10:
            print("  PASS: Critical instant R4 = 10 (matches analytical)")
        else:
            print(f"  MISMATCH: expected 10, got {first.response_time}")

    print()
    print("=== EDF Simulation (WCET) ===")
    result = simulate_edf(ts, use_wcet=True)
    print(result.summary())
