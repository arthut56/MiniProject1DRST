"""
Real-Time Scheduling Analysis Tool
==================================
Implements:
- Part A: Deadline Monotonic (DM) worst-case response times (RTA)
- Part B: EDF feasibility test (Processor Demand / DBF)
- Part C: EDF WCRT computation via schedule construction
- Part D: Discrete-event simulation for both DM and EDF
- Part E: Comparison and reporting

Task model: tau_i = (C_i, T_i, D_i, B_i)
- C_i: WCET (Worst-Case Execution Time)
- B_i: BCET (Best-Case ExecutiI want to hand all this over to claude. Can you explain everything it needs to do in latex in accordance to the chapteron Time) - used only for simulation
- T_i: Period
- D_i: Relative deadline (constrained: D_i <= T_i)

Synchronous release: all tasks released at time 0
"""

import math
from math import lcm
import heapq
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import random


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_hyperperiod(periods: List[int]) -> int:
    """Compute the hyperperiod (LCM) of all task periods."""
    H = 1
    for p in periods:
        H = lcm(H, int(p))
    return H


def compute_utilization(tasks: pd.DataFrame) -> float:
    """Compute total utilization U = sum(C_i / T_i)."""
    return float((tasks["WCET"] / tasks["Period"]).sum())


# =============================================================================
# PART A: DEADLINE MONOTONIC (DM) RTA
# =============================================================================

def dm_rta(tasks: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[bool, str]]:
    """
    Deadline Monotonic Response Time Analysis (RTA).

    Computes worst-case response times for each task under DM scheduling.
    Tasks are prioritized by increasing relative deadlines.

    Algorithm (Buttazzo Fig. 4.17):
    1. Sort tasks by non-decreasing D_i (DM priority order)
    2. For each task i, compute R_i iteratively:
       R_i^(0) = C_i
       R_i^(s) = C_i + sum_{h<i} ceil(R_i^(s-1) / T_h) * C_h
       Stop when converged or R_i > D_i (deadline miss)

    Returns:
        tasks: DataFrame with added "Ri_DM" column
        (schedulable, message): Tuple indicating schedulability
    """
    tasks = tasks.copy().reset_index(drop=True)

    # Sort by non-decreasing deadline (DM priority order)
    tasks = tasks.sort_values(by="Deadline").reset_index(drop=True)

    n = len(tasks)
    wcrts = []

    for i in range(n):
        Ci = int(tasks.loc[i, "WCET"])
        Di = int(tasks.loc[i, "Deadline"])

        # Initial response time estimate
        R = Ci

        max_iterations = 100000
        for iteration in range(max_iterations):
            R_old = R

            # Compute interference from higher-priority tasks (h < i)
            I = 0
            for h in range(i):
                Ch = int(tasks.loc[h, "WCET"])
                Th = int(tasks.loc[h, "Period"])
                I += math.ceil(R_old / Th) * Ch

            R = Ci + I

            # Check for deadline miss
            if R > Di:
                tasks["Ri_DM"] = wcrts + ["UNFEASIBLE"] * (n - i)
                return tasks, (False, f"Task {tasks.loc[i, 'Name']} misses deadline: R={R} > D={Di}")

            # Check for convergence
            if R == R_old:
                break

        wcrts.append(R)

    tasks["Ri_DM"] = wcrts
    return tasks, (True, "All tasks schedulable under DM")


def dm_schedulability_test(tasks: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """
    Complete DM schedulability test.

    Returns:
        (schedulable, message, tasks_with_wcrts)
    """
    tasks_result, (ok, msg) = dm_rta(tasks)
    return ok, msg, tasks_result


# =============================================================================
# PART B: EDF FEASIBILITY TEST (DBF)
# =============================================================================

def dbf(tasks: pd.DataFrame, t: int) -> int:
    """
    Demand-Bound Function for synchronous periodic tasks with D_i <= T_i.

    dbf(t) = sum_i floor((t + T_i - D_i) / T_i) * C_i
    """
    demand = 0
    for _, task in tasks.iterrows():
        C = int(task["WCET"])
        T = int(task["Period"])
        D = int(task["Deadline"])
        demand += math.floor((t + T - D) / T) * C
    return demand


def edf_dbf_feasibility_test(tasks: pd.DataFrame) -> Tuple[bool, str]:
    """
    EDF feasibility test using Demand-Bound Function (Chapter 4.6).

    Algorithm:
    1. If U >= 1, return infeasible
    2. Compute L* = sum((T_i - D_i) * U_i) / (1 - U)
    3. t_max = min(H, max(D_max, L*))
    4. Test dbf(t) <= t for all absolute deadlines t in D up to t_max

    Returns:
        (feasible, message)
    """
    tasks = tasks.copy().reset_index(drop=True)

    # Compute utilization
    U = compute_utilization(tasks)

    # Check for U approximately equal to 1.0 first (floating-point tolerance)
    if abs(U - 1.0) < 1e-9:
        # U == 1: feasible only if all deadlines == periods
        if (tasks["Deadline"] == tasks["Period"]).all():
            return True, f"Feasible: U=1.0 and all D_i = T_i"
        else:
            return False, f"Infeasible: U=1.0 but some D_i < T_i"

    if U > 1.0:
        return False, f"Infeasible: U={U:.6f} > 1"

    # Compute hyperperiod
    H = compute_hyperperiod(tasks["Period"].tolist())

    # Compute D_max
    D_max = int(tasks["Deadline"].max())

    # Compute L*
    sum_term = float(((tasks["Period"] - tasks["Deadline"]) * (tasks["WCET"] / tasks["Period"])).sum())
    L_star = sum_term / (1.0 - U)

    # Compute t_max
    t_max = min(H, max(D_max, int(math.ceil(L_star))))

    # Build set of absolute deadlines up to t_max
    test_points = set()
    for _, task in tasks.iterrows():
        T = int(task["Period"])
        D = int(task["Deadline"])
        k = 0
        while True:
            d = D + k * T
            if d > t_max:
                break
            if d > 0:
                test_points.add(d)
            k += 1

    # Test DBF at each deadline
    for t in sorted(test_points):
        demand = dbf(tasks, t)
        if demand > t:
            return False, f"Infeasible at t={t}: dbf(t)={demand} > {t}"

    return True, f"Feasible by DBF test (U={U:.6f}, tested up to t_max={t_max})"


# =============================================================================
# PART C: EDF WCRT COMPUTATION (Schedule Construction)
# =============================================================================

@dataclass(order=True)
class Job:
    """Represents a job instance in the schedule."""
    priority: float = field(compare=True)  # For heap ordering
    task_id: int = field(compare=False)
    job_id: int = field(compare=False)
    release_time: int = field(compare=False)
    absolute_deadline: int = field(compare=False)
    execution_time: int = field(compare=False)
    remaining_time: int = field(compare=False)
    start_time: Optional[int] = field(compare=False, default=None)
    finish_time: Optional[int] = field(compare=False, default=None)


def edf_wcrt_schedule_construction(tasks: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[bool, str]]:
    """
    Compute EDF WCRTs by constructing the deterministic WCET schedule.

    Simulates one hyperperiod with all jobs executing for their WCET.
    Uses event-driven simulation with preemptive EDF.

    Returns:
        tasks: DataFrame with added "Ri_EDF" column
        (ok, message): Status tuple
    """
    tasks = tasks.copy().reset_index(drop=True)

    # First check feasibility
    ok, msg = edf_dbf_feasibility_test(tasks)
    if not ok:
        tasks["Ri_EDF"] = ["UNFEASIBLE"] * len(tasks)
        return tasks, (False, msg)

    n = len(tasks)
    H = compute_hyperperiod(tasks["Period"].tolist())

    # Safety limit for very large hyperperiods
    max_sim_time = min(H, 10**8)

    # Generate all jobs released in [0, H)
    jobs = []  # List of all jobs
    for i in range(n):
        Ti = int(tasks.loc[i, "Period"])
        Di = int(tasks.loc[i, "Deadline"])
        Ci = int(tasks.loc[i, "WCET"])

        k = 0
        while k * Ti < max_sim_time:
            r = k * Ti
            d = r + Di
            job = Job(
                priority=d,  # EDF: priority by absolute deadline
                task_id=i,
                job_id=k,
                release_time=r,
                absolute_deadline=d,
                execution_time=Ci,
                remaining_time=Ci
            )
            jobs.append(job)
            k += 1

    # Sort jobs by release time for event generation
    jobs.sort(key=lambda j: (j.release_time, j.absolute_deadline))

    # Event-driven simulation
    current_time = 0
    ready_queue = []  # Min-heap by absolute deadline
    job_index = 0  # Next job to release
    running_job = None
    completed_jobs = []

    max_C = int(tasks["WCET"].max())
    end_time = max_sim_time + max_C

    while current_time < end_time:
        # Release all jobs at current_time
        while job_index < len(jobs) and jobs[job_index].release_time == current_time:
            job = jobs[job_index]
            heapq.heappush(ready_queue, (job.absolute_deadline, job.task_id, job.job_id, job))
            job_index += 1

        # Select job with earliest deadline
        if ready_queue:
            _, _, _, best_job = ready_queue[0]

            # Preemption check
            if running_job is not None and running_job is not best_job:
                # Preempt current job (it's already in ready_queue)
                pass

            running_job = best_job
        else:
            running_job = None

        # Find next event time
        next_release = jobs[job_index].release_time if job_index < len(jobs) else float('inf')

        if running_job is not None:
            # Run until completion or next release
            run_until = min(current_time + running_job.remaining_time, next_release)
            exec_time = run_until - current_time

            if running_job.start_time is None:
                running_job.start_time = current_time

            running_job.remaining_time -= exec_time
            current_time = run_until

            if running_job.remaining_time == 0:
                running_job.finish_time = current_time
                completed_jobs.append(running_job)
                # Remove from ready queue
                ready_queue = [(d, ti, ji, j) for (d, ti, ji, j) in ready_queue if j is not running_job]
                heapq.heapify(ready_queue)
                running_job = None
        else:
            # Idle - jump to next release
            if next_release == float('inf'):
                break
            current_time = int(next_release)

    # Compute WCRT for each task (max response time of jobs released before H)
    wcrts = [0] * n
    for job in completed_jobs:
        if job.release_time < H:
            R = job.finish_time - job.release_time
            if R > job.absolute_deadline - job.release_time:
                # Deadline miss detected
                tasks["Ri_EDF"] = ["UNFEASIBLE"] * n
                return tasks, (False, f"Task {job.task_id} job {job.job_id} missed deadline")
            wcrts[job.task_id] = max(wcrts[job.task_id], R)

    tasks["Ri_EDF"] = wcrts
    return tasks, (True, "EDF WCRT computed via schedule construction")


# =============================================================================
# PART D: DISCRETE-EVENT SIMULATION
# =============================================================================

class EventType(Enum):
    RELEASE = 1
    COMPLETE = 2


@dataclass
class Event:
    """Simulation event."""
    time: int
    event_type: EventType
    task_id: int
    job_id: int


@dataclass
class SimJob:
    """Job instance for simulation."""
    task_id: int
    job_id: int
    release_time: int
    absolute_deadline: int
    execution_time: int  # Sampled execution time (BCET to WCET)
    remaining_time: int
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    preemption_count: int = 0


class DiscreteEventSimulator:
    """
    Discrete-event simulator for DM and EDF scheduling.

    Uses event-scheduling approach:
    - Events: RELEASE, COMPLETE
    - State: current time, ready queue, running job, per-job timestamps
    """

    def __init__(self, tasks: pd.DataFrame, policy: str = "EDF",
                 use_wcet: bool = False, seed: Optional[int] = None):
        """
        Initialize simulator.

        Args:
            tasks: Task set DataFrame
            policy: "EDF" or "DM"
            use_wcet: If True, use WCET; else sample from [BCET, WCET]
            seed: Random seed for reproducibility
        """
        self.tasks = tasks.copy().reset_index(drop=True)
        self.policy = policy
        self.use_wcet = use_wcet
        self.rng = random.Random(seed)
        self.n = len(tasks)

        # Precompute task parameters
        self.task_params = []
        for i in range(self.n):
            self.task_params.append({
                'C': int(tasks.loc[i, "WCET"]),
                'B': int(tasks.loc[i, "BCET"]) if "BCET" in tasks.columns else int(tasks.loc[i, "WCET"]),
                'T': int(tasks.loc[i, "Period"]),
                'D': int(tasks.loc[i, "Deadline"])
            })

        # DM priorities (lower = higher priority, by deadline)
        if policy == "DM":
            deadlines = [self.task_params[i]['D'] for i in range(self.n)]
            sorted_indices = sorted(range(self.n), key=lambda i: deadlines[i])
            self.dm_priority = {idx: rank for rank, idx in enumerate(sorted_indices)}

    def sample_execution_time(self, task_id: int) -> int:
        """Sample execution time from [BCET, WCET] uniformly."""
        if self.use_wcet:
            return self.task_params[task_id]['C']
        B = self.task_params[task_id]['B']
        C = self.task_params[task_id]['C']
        return self.rng.randint(B, C)

    def get_priority(self, job: SimJob) -> Tuple:
        """
        Get priority tuple for job ordering (lower = higher priority).

        For DM: (dm_priority[task_id], release_time)
        For EDF: (absolute_deadline, release_time)
        """
        if self.policy == "DM":
            return (self.dm_priority[job.task_id], job.release_time)
        else:  # EDF
            return (job.absolute_deadline, job.release_time)

    def run(self, simulation_time: int) -> Dict[str, Any]:
        """
        Run simulation for specified time.

        Returns:
            Dictionary with:
            - 'response_times': {task_id: [list of response times]}
            - 'max_response_times': {task_id: max_R}
            - 'deadline_misses': {task_id: count}
            - 'preemptions': {task_id: total_preemptions}
            - 'completed_jobs': list of all completed SimJob
        """
        # Initialize state
        current_time = 0
        event_queue = []  # Min-heap of (time, event_type_value, task_id, job_id)
        ready_queue = []  # List of SimJob, managed manually
        running_job: Optional[SimJob] = None
        jobs_by_id: Dict[Tuple[int, int], SimJob] = {}

        # Results
        response_times = {i: [] for i in range(self.n)}
        deadline_misses = {i: 0 for i in range(self.n)}
        preemptions = {i: 0 for i in range(self.n)}
        completed_jobs = []

        # Schedule initial releases (all at time 0 for synchronous release)
        for i in range(self.n):
            heapq.heappush(event_queue, (0, EventType.RELEASE.value, i, 0))

        while event_queue and current_time <= simulation_time:
            event_time, event_type_val, task_id, job_id = heapq.heappop(event_queue)

            # Execute running job up to event time
            if running_job is not None and event_time > current_time:
                exec_time = event_time - current_time
                running_job.remaining_time -= exec_time

            current_time = event_time

            if event_type_val == EventType.RELEASE.value:
                # Create new job
                T = self.task_params[task_id]['T']
                D = self.task_params[task_id]['D']
                exec_time = self.sample_execution_time(task_id)

                job = SimJob(
                    task_id=task_id,
                    job_id=job_id,
                    release_time=current_time,
                    absolute_deadline=current_time + D,
                    execution_time=exec_time,
                    remaining_time=exec_time
                )
                jobs_by_id[(task_id, job_id)] = job
                ready_queue.append(job)

                # Schedule next release
                next_release = current_time + T
                if next_release <= simulation_time:
                    heapq.heappush(event_queue, (next_release, EventType.RELEASE.value, task_id, job_id + 1))

                # Check for preemption
                if running_job is not None:
                    job_priority = self.get_priority(job)
                    running_priority = self.get_priority(running_job)
                    if job_priority < running_priority:
                        # Preempt running job
                        running_job.preemption_count += 1
                        preemptions[running_job.task_id] += 1
                        running_job = None

            elif event_type_val == EventType.COMPLETE.value:
                # Job completion
                job = jobs_by_id.get((task_id, job_id))
                if job is not None and job.remaining_time == 0:
                    job.finish_time = current_time
                    R = job.finish_time - job.release_time
                    response_times[task_id].append(R)
                    completed_jobs.append(job)

                    if R > self.task_params[task_id]['D']:
                        deadline_misses[task_id] += 1

                    # Remove from ready queue
                    ready_queue = [j for j in ready_queue if j is not job]

                    if running_job is job:
                        running_job = None

            # Select next job to run
            if ready_queue:
                ready_queue.sort(key=lambda j: self.get_priority(j))
                next_job = ready_queue[0]

                if running_job is None:
                    running_job = next_job
                    if running_job.start_time is None:
                        running_job.start_time = current_time

                    # Schedule completion
                    finish_time = current_time + running_job.remaining_time
                    heapq.heappush(event_queue,
                                   (finish_time, EventType.COMPLETE.value,
                                    running_job.task_id, running_job.job_id))

                elif next_job is not running_job:
                    # Check if we need to switch
                    if self.get_priority(next_job) < self.get_priority(running_job):
                        # Preempt and switch
                        running_job.preemption_count += 1
                        preemptions[running_job.task_id] += 1
                        running_job = next_job
                        if running_job.start_time is None:
                            running_job.start_time = current_time

                        # Schedule completion
                        finish_time = current_time + running_job.remaining_time
                        heapq.heappush(event_queue,
                                       (finish_time, EventType.COMPLETE.value,
                                        running_job.task_id, running_job.job_id))

        # Compute max response times
        max_response_times = {}
        for i in range(self.n):
            if response_times[i]:
                max_response_times[i] = max(response_times[i])
            else:
                max_response_times[i] = None

        return {
            'response_times': response_times,
            'max_response_times': max_response_times,
            'deadline_misses': deadline_misses,
            'preemptions': preemptions,
            'completed_jobs': completed_jobs
        }


def simulate_schedule(tasks: pd.DataFrame, policy: str = "DM",
                      use_wcet: bool = True, max_sim_time: int = 100000,
                      seed: int = 42) -> Dict[str, Any]:
    """
    Run discrete-event simulation for DM or EDF scheduling.

    Args:
        tasks: Task set DataFrame
        policy: "DM" or "EDF"
        use_wcet: If True, use WCET (deterministic worst-case simulation)
                  If False, sample from [BCET, WCET]
        max_sim_time: Maximum simulation time
        seed: Random seed (only used if use_wcet=False)

    Returns:
        Dictionary with simulation results
    """
    H = compute_hyperperiod(tasks["Period"].tolist())
    sim_time = min(H, max_sim_time)

    sim = DiscreteEventSimulator(tasks, policy=policy, use_wcet=use_wcet, seed=seed)
    result = sim.run(sim_time)

    return {
        'max_response_times': result['max_response_times'],
        'response_times': result['response_times'],
        'deadline_misses': result['deadline_misses'],
        'preemptions': result['preemptions'],
        'sim_time': sim_time,
        'use_wcet': use_wcet
    }


# =============================================================================
# PART E: COMPARISON AND REPORTING
# =============================================================================

def analyze_task_set(tasks: pd.DataFrame, num_sim_runs: int = 100,
                     num_hyperperiods: int = 1, seed: int = 42) -> Dict[str, Any]:
    """
    Complete analysis of a task set.

    Performs:
    1. Utilization computation
    2. DM RTA and schedulability
    3. EDF DBF feasibility test
    4. EDF WCRT computation (schedule construction)
    5. Discrete-event simulation for both DM and EDF (WCET-based)
    6. Comparison of analytical vs simulated results

    Returns comprehensive analysis results.
    """
    print("=" * 70)
    print("REAL-TIME SCHEDULING ANALYSIS")
    print("=" * 70)

    n = len(tasks)

    # 1. Utilization
    U = compute_utilization(tasks)
    print(f"\n1. UTILIZATION: U = {U:.6f}")

    # 2. DM RTA
    print("\n2. DEADLINE MONOTONIC (DM) ANALYSIS")
    print("-" * 40)
    dm_ok, dm_msg, tasks_dm = dm_schedulability_test(tasks)
    print(f"   Schedulable: {dm_ok}")
    print(f"   Message: {dm_msg}")

    # 3. EDF DBF Feasibility
    print("\n3. EDF FEASIBILITY (DBF TEST)")
    print("-" * 40)
    edf_ok, edf_msg = edf_dbf_feasibility_test(tasks)
    print(f"   Feasible: {edf_ok}")
    print(f"   Message: {edf_msg}")

    # 4. EDF WCRT (Schedule Construction)
    print("\n4. EDF WCRT (SCHEDULE CONSTRUCTION)")
    print("-" * 40)
    tasks_edf, (edf_wcrt_ok, edf_wcrt_msg) = edf_wcrt_schedule_construction(tasks)
    print(f"   Success: {edf_wcrt_ok}")
    print(f"   Message: {edf_wcrt_msg}")

    # Combine results
    results_df = tasks.copy()

    # Add DM results (need to align by task name since DM sorts by deadline)
    dm_results = tasks_dm[["Name", "Ri_DM"]].set_index("Name")
    results_df = results_df.set_index("Name")
    results_df["Ri_DM"] = dm_results["Ri_DM"]
    results_df = results_df.reset_index()

    # Add EDF results
    results_df["Ri_EDF"] = tasks_edf["Ri_EDF"].values

    print("\n   Per-task Analytical WCRTs:")
    print("   " + "-" * 50)
    print(f"   {'Task':<8} {'D_i':<8} {'Ri_DM':<12} {'Ri_EDF':<12}")
    print("   " + "-" * 50)
    for i, row in results_df.iterrows():
        name = row['Name']
        Di = row['Deadline']
        Ri_DM = row['Ri_DM']
        Ri_EDF = row['Ri_EDF']
        print(f"   {name:<8} {Di:<8} {str(Ri_DM):<12} {str(Ri_EDF):<12}")

    # 5. Discrete-Event Simulation (WCET-based for comparison with analysis)
    print("\n5. DISCRETE-EVENT SIMULATION (WCET)")
    print("-" * 40)

    H = compute_hyperperiod(tasks["Period"].tolist())
    print(f"   Hyperperiod H = {H}")

    max_sim_time = min(H, 100000)
    if H > max_sim_time:
        print(f"   WARNING: Hyperperiod large, using simulation time = {max_sim_time}")

    print("\n   Running DM simulation (WCET)...")
    dm_sim = simulate_schedule(tasks, policy="DM", use_wcet=True, max_sim_time=max_sim_time)

    print("   Running EDF simulation (WCET)...")
    edf_sim = simulate_schedule(tasks, policy="EDF", use_wcet=True, max_sim_time=max_sim_time)

    # Add simulation results to DataFrame
    results_df["Ri_DM_sim"] = [dm_sim['max_response_times'].get(i, None) for i in range(n)]
    results_df["Ri_EDF_sim"] = [edf_sim['max_response_times'].get(i, None) for i in range(n)]
    results_df["DM_preemptions"] = [dm_sim['preemptions'].get(i, 0) for i in range(n)]
    results_df["EDF_preemptions"] = [edf_sim['preemptions'].get(i, 0) for i in range(n)]

    # 6. Comparison: Analytical vs Simulation
    print("\n6. COMPARISON: ANALYTICAL vs SIMULATION")
    print("-" * 70)
    print(f"   {'Task':<8} {'D_i':<6} {'DM_ana':<8} {'DM_sim':<8} {'Match':<6} {'EDF_ana':<8} {'EDF_sim':<8} {'Match':<6}")
    print("   " + "-" * 70)

    dm_matches = 0
    edf_matches = 0
    for i, row in results_df.iterrows():
        name = row['Name']
        Di = row['Deadline']
        dm_ana = row['Ri_DM']
        dm_sim_val = row['Ri_DM_sim']
        edf_ana = row['Ri_EDF']
        edf_sim_val = row['Ri_EDF_sim']

        # Check if analytical matches simulation
        dm_match = "✓" if (dm_ana == dm_sim_val or (isinstance(dm_ana, str) and dm_ana == "UNFEASIBLE")) else "✗"
        edf_match = "✓" if (edf_ana == edf_sim_val or (isinstance(edf_ana, str) and edf_ana == "UNFEASIBLE")) else "✗"

        if dm_match == "✓":
            dm_matches += 1
        if edf_match == "✓":
            edf_matches += 1

        print(f"   {name:<8} {Di:<6} {str(dm_ana):<8} {str(dm_sim_val):<8} {dm_match:<6} {str(edf_ana):<8} {str(edf_sim_val):<8} {edf_match:<6}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Total Utilization: {U:.6f}")
    print(f"   DM Schedulable (analytical): {dm_ok}")
    print(f"   EDF Feasible (analytical): {edf_ok}")
    print(f"\n   DM: Analytical vs Simulation match: {dm_matches}/{n} tasks")
    print(f"   EDF: Analytical vs Simulation match: {edf_matches}/{n} tasks")

    dm_total_misses = sum(dm_sim['deadline_misses'].values())
    edf_total_misses = sum(edf_sim['deadline_misses'].values())
    print(f"\n   DM Deadline Misses (simulation): {dm_total_misses}")
    print(f"   EDF Deadline Misses (simulation): {edf_total_misses}")

    dm_total_preempt = sum(dm_sim['preemptions'].values())
    edf_total_preempt = sum(edf_sim['preemptions'].values())
    print(f"   DM Total Preemptions: {dm_total_preempt}")
    print(f"   EDF Total Preemptions: {edf_total_preempt}")

    return {
        'utilization': U,
        'dm_schedulable': dm_ok,
        'dm_message': dm_msg,
        'edf_feasible': edf_ok,
        'edf_message': edf_msg,
        'results_df': results_df,
        'dm_sim': dm_sim,
        'edf_sim': edf_sim
    }


def generate_report(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Generate a formatted report from analysis results."""

    lines = []
    lines.append("=" * 70)
    lines.append("REAL-TIME SCHEDULING ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Total Utilization: U = {results['utilization']:.6f}")
    lines.append("")

    lines.append("DEADLINE MONOTONIC (DM) ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Schedulable: {results['dm_schedulable']}")
    lines.append(f"Message: {results['dm_message']}")
    lines.append("")

    lines.append("EDF ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Feasible (DBF): {results['edf_feasible']}")
    lines.append(f"Message: {results['edf_message']}")
    lines.append("")

    lines.append("PER-TASK RESULTS")
    lines.append("-" * 40)
    df = results['results_df']
    lines.append(df.to_string())
    lines.append("")

    lines.append("SIMULATION STATISTICS")
    lines.append("-" * 40)
    lines.append(f"DM Total Deadline Misses: {sum(results['dm_sim']['deadline_misses'].values())}")
    lines.append(f"EDF Total Deadline Misses: {sum(results['edf_sim']['deadline_misses'].values())}")
    lines.append(f"DM Total Preemptions: {sum(results['dm_sim']['preemptions'].values())}")
    lines.append(f"EDF Total Preemptions: {sum(results['edf_sim']['preemptions'].values())}")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)

    return report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different CSV formats."""
    tasks = tasks.copy()
    # Rename 'Task' to 'Name' if needed
    if 'Task' in tasks.columns and 'Name' not in tasks.columns:
        tasks = tasks.rename(columns={'Task': 'Name'})
    return tasks


if __name__ == "__main__":
    # Load task set
    csv_path = 'task_sets/unschedulable/Unschedulable_Full_Utilization_NonUnique_Periods_taskset.csv'
    print(f"Loading task set from {csv_path}...")
    tasks = pd.read_csv(csv_path)
    tasks = normalize_task_columns(tasks)

    print(f"\nTask Set ({len(tasks)} tasks):")
    print(tasks[['Name', 'BCET', 'WCET', 'Period', 'Deadline']].to_string())

    # Run complete analysis
    results = analyze_task_set(
        tasks,
        num_sim_runs=10,  # Reduced for faster testing
        num_hyperperiods=1,
        seed=42
    )

    # Save results to CSV
    results['results_df'].to_csv('data/analysis_results.csv', index=False)
    print("\nResults saved to data/analysis_results.csv")

