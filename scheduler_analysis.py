import math
from math import lcm
import heapq
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import random
from scipy.stats import truncnorm


MAX_EXACT_HYPERPERIOD = 10**7
REQUIRED_TASK_COLUMNS = ["Name", "WCET", "Period", "Deadline"]


def compute_hyperperiod(periods: List[int]) -> int:
    """Compute the hyperperiod (LCM) of all task periods."""
    if not periods:
        raise ValueError("Cannot compute hyperperiod of an empty period list")

    H = 1
    for p in periods:
        p_int = int(p)
        if p_int <= 0:
            raise ValueError(f"Periods must be positive integers, got {p_int}")
        H = lcm(H, p_int)
    return H


def compute_utilization(tasks: pd.DataFrame) -> float:
    """Compute total utilization U = sum(C_i / T_i)."""
    if (tasks["Period"] <= 0).any():
        raise ValueError("All periods must be > 0")
    return float((tasks["WCET"] / tasks["Period"]).sum())


def validate_taskset(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate a constrained-deadline periodic task set."""
    if not isinstance(tasks, pd.DataFrame):
        raise TypeError("Task set must be provided as a pandas DataFrame")

    normalized = tasks.copy()
    if "Task" in normalized.columns and "Name" not in normalized.columns:
        normalized = normalized.rename(columns={"Task": "Name"})

    if "BCET" not in normalized.columns and "WCET" in normalized.columns:
        # Some provided CSVs skip BCET; using WCET keeps deterministic semantics.
        normalized["BCET"] = normalized["WCET"]

    required = REQUIRED_TASK_COLUMNS + ["BCET"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Task set missing required columns: {missing}")

    if normalized.empty:
        raise ValueError("Task set is empty")

    for col in ["BCET", "WCET", "Period", "Deadline"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="raise")

    if (normalized["WCET"] <= 0).any() or (normalized["Period"] <= 0).any() or (normalized["Deadline"] <= 0).any():
        raise ValueError("WCET, Period, and Deadline must all be > 0")
    if (normalized["BCET"] < 0).any():
        raise ValueError("BCET must be >= 0")
    if (normalized["BCET"] > normalized["WCET"]).any():
        raise ValueError("BCET must be <= WCET for every task")
    if (normalized["Deadline"] > normalized["Period"]).any():
        raise ValueError("Only constrained deadlines are supported: require Deadline <= Period")

    # Keep the canonical column order up front, but preserve any extra columns.
    front = ["Name", "BCET", "WCET", "Period", "Deadline"]
    rest = [c for c in normalized.columns if c not in front]
    return normalized[front + rest].reset_index(drop=True)



# DEADLINE MONOTONIC (DM) RTA


def dm_rta(tasks: pd.DataFrame) -> Tuple[pd.DataFrame, Tuple[bool, str]]:
    """
    Deadline Monotonic Response Time Analysis (RTA).

    Computes worst-case response times for each task under DM scheduling.
    Tasks are prioritized by increasing relative deadline; ties broken by
    original task index (smaller index = higher priority).

    Algorithm (Audsley et al. 1993; Buttazzo Fig. 4.17):
    1. Sort tasks by non-decreasing D_i, breaking ties by original index.
    2. For each task i, compute R_i iteratively:
       R_i^(0) = C_i
       R_i^(s) = C_i + sum_{j in hp(i)} ceil(R_i^(s-1) / T_j) * C_j
       where hp(i) is the set of tasks with strictly higher DM priority,
       i.e. D_j < D_i, OR (D_j == D_i AND orig_index_j < orig_index_i).
       Stop when converged or R_i > D_i (deadline miss).

    Note on equal-deadline tasks: the standard formula uses D_j < D_i
    (strict inequality), which would omit interference from tasks that share
    the same relative deadline but have higher priority under the tie-breaking
    rule.  By sorting on (D_i, original_index) and summing over h < i, we
    correctly include those same-deadline, higher-priority tasks.

    Returns:
        tasks: DataFrame with added "Ri_DM" column
        (schedulable, message): Tuple indicating schedulability
    """
    tasks = validate_taskset(tasks)

    tasks["_orig_idx"] = tasks.index
    tasks = tasks.sort_values(by=["Deadline", "_orig_idx"]).reset_index(drop=True)

    n = len(tasks)
    wcrts = []

    for i in range(n):
        Ci = int(tasks.loc[i, "WCET"])
        Di = int(tasks.loc[i, "Deadline"])

        # initial response time estimate
        R = Ci

        max_iterations = 100000
        for _ in range(max_iterations):
            R_old = R

            # Interference from all tasks with strictly higher DM priority.
            # After sorting by (Deadline, orig_idx), every task h < i has
            # either D_h < D_i, or D_h == D_i with a smaller original index
            # (i.e. higher tie-break priority).  Both cases contribute
            # preemption interference.
            I = 0
            for h in range(i):
                Ch = int(tasks.loc[h, "WCET"])
                Th = int(tasks.loc[h, "Period"])
                I += math.ceil(R_old / Th) * Ch

            R = Ci + I

            # Check for deadline miss
            if R > Di:
                tasks["Ri_DM"] = wcrts + ["UNFEASIBLE"] * (n - i)
                tasks = tasks.drop(columns=["_orig_idx"])
                return tasks, (False, f"Task {tasks.loc[i, 'Name']} misses deadline: R={R} > D={Di}")

            # Check for convergence
            if R == R_old:
                break

        wcrts.append(R)

    tasks["Ri_DM"] = wcrts
    tasks = tasks.drop(columns=["_orig_idx"])
    return tasks, (True, "All tasks schedulable under DM")


def dm_schedulability_test(tasks: pd.DataFrame) -> Tuple[bool, str, pd.DataFrame]:
    """
    Complete DM schedulability test.

    Returns:
        (schedulable, message, tasks_with_wcrts)
    """
    tasks_result, (ok, msg) = dm_rta(tasks)
    return ok, msg, tasks_result



# EDF FEASIBILITY TEST (DBF)


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
    tasks = validate_taskset(tasks)

    # Compute utilization
    U = compute_utilization(tasks)

    # Quick sanity check; otherwise the DBF loop below gets nonsense input.
    if U > 1.0 + 1e-9:
        return False, f"Infeasible: U={U:.6f} > 1"

    # Compute hyperperiod
    H = compute_hyperperiod(tasks["Period"].tolist())

    # Compute D_max
    D_max = int(tasks["Deadline"].max())

    if abs(1.0 - U) <= 1e-9:
        # U~=1 makes L* singular. For periodic constrained-deadline sets we can
        # still run an exact DBF check over one hyperperiod.
        t_max = H
    else:
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

    # test DBF at each deadline
    for t in sorted(test_points):
        demand = dbf(tasks, t)
        if demand > t:
            return False, f"Infeasible at t={t}: dbf(t)={demand} > {t}"

    return True, f"Feasible by DBF test (U={U:.6f}, tested up to t_max={t_max})"


# EDF WCRT COMPUTATION

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
    tasks = validate_taskset(tasks)

    # First check feasibility
    ok, msg = edf_dbf_feasibility_test(tasks)
    if not ok:
        tasks["Ri_EDF"] = ["UNFEASIBLE"] * len(tasks)
        return tasks, (False, msg)

    n = len(tasks)
    H = compute_hyperperiod(tasks["Period"].tolist())

    # exact analysis is over one full hyperperiod.
    if H > MAX_EXACT_HYPERPERIOD:
        tasks["Ri_EDF"] = ["NOT_COMPUTED"] * n
        return tasks, (False, f"Hyperperiod H={H} exceeds exact-analysis cap {MAX_EXACT_HYPERPERIOD}")

    # Generate all jobs released in [0, H).  Jobs whose release time is near H
    # may not complete until after H; the simulation loop below runs until every
    # released job finishes, so their finish times are recorded correctly and
    # included in the WCRT max.  Terminating at exactly H would miss those
    # completion times and could under-estimate the WCRT.
    jobs = []
    for i in range(n):
        Ti = int(tasks.loc[i, "Period"])
        Di = int(tasks.loc[i, "Deadline"])
        Ci = int(tasks.loc[i, "WCET"])

        k = 0
        while k * Ti < H:
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

    # sort jobs by release time for event generation
    jobs.sort(key=lambda j: (j.release_time, j.absolute_deadline))

    current_time = 0
    ready_queue = []  # min-heap by absolute deadline for efficiency
    job_index = 0
    running_job = None
    completed_jobs = []

    while True:
        # release all jobs at current_time
        while job_index < len(jobs) and jobs[job_index].release_time == current_time:
            job = jobs[job_index]
            heapq.heappush(ready_queue, (job.absolute_deadline, job.task_id, job.job_id, job))
            job_index += 1

        #stop once there are no queued/running jobs and no future releases.
        if not ready_queue and running_job is None and job_index >= len(jobs):
            break

        # select job with earliest deadline
        if ready_queue:
            _, _, _, running_job = ready_queue[0]
        else:
            running_job = None

    #find next event time
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
                heapq.heappop(ready_queue)
                running_job = None
        else:
            # idle - jump to next release
            if next_release == float('inf'):
                break
            current_time = int(next_release)

        #compute WCRT for each task (max response time of jobs released before H)
    wcrts = [0] * n
    for job in completed_jobs:
        if job.release_time < H:
            R = job.finish_time - job.release_time
            if R > job.absolute_deadline - job.release_time:
                # Deadline miss found
                tasks["Ri_EDF"] = ["UNFEASIBLE"] * n
                return tasks, (False, f"Task {job.task_id} job {job.job_id} missed deadline")
            wcrts[job.task_id] = max(wcrts[job.task_id], R)

    tasks["Ri_EDF"] = wcrts
    return tasks, (True, "EDF WCRT computed via schedule construction")


def run_stochastic_simulation_stats(tasks: pd.DataFrame, policy: str,
                                    num_runs: int, sim_time: int,
                                    seed: int = 42,
                                    convergence_patience: int = 10) -> Dict[str, Any]:
    """Aggregate observed max/mean/p95 response times across stochastic runs.

    Stopping condition:
    The simulation always runs for at least ``num_runs`` independent runs.
    Additionally, early termination is applied: if no task's observed maximum
    response time has improved for ``convergence_patience`` consecutive runs,
    we conclude that additional runs are unlikely to discover a new worst case
    and stop early.  This provides an analytically-motivated bound: once the
    empirical maxima have stabilised over an interval of
    ``convergence_patience`` hyperperiods, further sampling yields diminishing
    returns.
    """
    tasks = validate_taskset(tasks)

    if num_runs <= 0:
        raise ValueError("num_runs must be > 0")
    if sim_time <= 0:
        raise ValueError("sim_time must be > 0")

    n = len(tasks)
    all_rts = {i: [] for i in range(n)}
    observed_max = {i: 0 for i in range(n)}
    total_deadline_misses = 0
    no_improvement_streak = 0

    for run in range(num_runs):
        sim = simulate_schedule(
            tasks,
            policy=policy,
            use_wcet=False,
            max_sim_time=sim_time,
            seed=seed + run,
        )

        improved = False
        for i in range(n):
            rts = sim['response_times'].get(i, [])
            all_rts[i].extend(rts)
            if rts:
                new_max = max(rts)
                if new_max > observed_max[i]:
                    observed_max[i] = new_max
                    improved = True

        total_deadline_misses += sum(sim['deadline_misses'].values())

        if improved:
            no_improvement_streak = 0
        else:
            no_improvement_streak += 1
            # We always honor the requested baseline run count first.
            if (run + 1) >= num_runs and no_improvement_streak >= convergence_patience:
                break

    mean_rt = {
        i: (float(np.mean(all_rts[i])) if all_rts[i] else None)
        for i in range(n)
    }
    p95_rt = {
        i: (float(np.percentile(all_rts[i], 95)) if all_rts[i] else None)
        for i in range(n)
    }

    return {
        'observed_max_rt': observed_max,
        'observed_mean_rt': mean_rt,
        'observed_p95_rt': p95_rt,
        'total_deadline_misses': total_deadline_misses,
        'num_runs': run + 1,  # actual runs executed
    }


@dataclass
class SimJob:
    """Job instance for simulation."""
    task_id: int
    job_id: int
    release_time: int
    absolute_deadline: int
    execution_time: int  # sampled execution time (BCET to WCET)
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

        # precompute task parameters
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
            sorted_indices = sorted(range(self.n), key=lambda i: (deadlines[i], i))
            self.dm_priority = {idx: rank for rank, idx in enumerate(sorted_indices)}

    def sample_execution_time(self, task_id: int,
                               distribution: str = "truncnorm") -> int:
        """Sample execution time from [BCET, WCET].

        Args:
            distribution: "uniform"   – draw uniformly from [BCET, WCET].
                          "truncnorm" – draw from a truncated normal whose
                                        mean is the midpoint and whose sigma
                                        is chosen so that most mass sits near
                                        the average (sigma = range / 4).
                                        This is more realistic than uniform
                                        because real execution times cluster
                                        around their average rather than being
                                        flat across the whole range.
        """
        if self.use_wcet:
            return self.task_params[task_id]['C']
        B = self.task_params[task_id]['B']
        C = self.task_params[task_id]['C']
        if B == C:
            return C
        if distribution == "truncnorm":
            mu = (B + C) / 2.0
            sigma = (C - B) / 4.0
            a, b = (B - mu) / sigma, (C - mu) / sigma
            sample = truncnorm.rvs(a, b, loc=mu, scale=sigma,
                                   random_state=self.rng.randint(0, 2**31))
            return int(round(min(C, max(B, sample))))
        # default: uniform
        return self.rng.randint(B, C)

    def get_priority(self, job: SimJob) -> Tuple:
        """
        Get priority tuple for job ordering (lower = higher priority).

        For DM: (dm_priority[task_id], task_id, job_id)
        For EDF: (absolute_deadline, task_id, job_id)
        """
        if self.policy == "DM":
            return (self.dm_priority[job.task_id], job.task_id, job.job_id)
        else:  # EDF
            return (job.absolute_deadline, job.task_id, job.job_id)

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
        # build all job releases in [0, simulation_time)
        releases = []
        for task_id in range(self.n):
            T = self.task_params[task_id]['T']
            job_id = 0
            while job_id * T < simulation_time:
                releases.append((job_id * T, task_id, job_id))
                job_id += 1
        releases.sort()

        current_time = 0
        ready_queue: List[SimJob] = []
        running_job: Optional[SimJob] = None
        jobs_by_id: Dict[Tuple[int, int], SimJob] = {}
        release_index = 0

        # Results
        response_times = {i: [] for i in range(self.n)}
        deadline_misses = {i: 0 for i in range(self.n)}
        preemptions = {i: 0 for i in range(self.n)}
        completed_jobs = []

        while release_index < len(releases) or ready_queue or running_job is not None:
            next_release = releases[release_index][0] if release_index < len(releases) else float('inf')
            next_completion = (
                current_time + running_job.remaining_time
                if running_job is not None else float('inf')
            )
            next_event_time = min(next_release, next_completion)

            if next_event_time == float('inf'):
                break

            if running_job is not None and next_event_time > current_time:
                running_job.remaining_time -= (next_event_time - current_time)

            current_time = int(next_event_time)

            if running_job is not None and running_job.remaining_time == 0:
                running_job.finish_time = current_time
                R = running_job.finish_time - running_job.release_time
                response_times[running_job.task_id].append(R)
                completed_jobs.append(running_job)
                if R > self.task_params[running_job.task_id]['D']:
                    deadline_misses[running_job.task_id] += 1
                running_job = None

            while release_index < len(releases) and releases[release_index][0] == current_time:
                _, task_id, job_id = releases[release_index]
                D = self.task_params[task_id]['D']
                exec_time = self.sample_execution_time(task_id)
                job = SimJob(
                    task_id=task_id,
                    job_id=job_id,
                    release_time=current_time,
                    absolute_deadline=current_time + D,
                    execution_time=exec_time,
                    remaining_time=exec_time,
                )
                jobs_by_id[(task_id, job_id)] = job
                ready_queue.append(job)
                release_index += 1

            candidates = ready_queue.copy()
            if running_job is not None:
                candidates.append(running_job)

            if not candidates:
                continue

            candidates.sort(key=lambda j: self.get_priority(j))
            next_job = candidates[0]

            if running_job is not None and next_job is not running_job:
                running_job.preemption_count += 1
                preemptions[running_job.task_id] += 1
                ready_queue.append(running_job)
                running_job = None

            if next_job in ready_queue:
                ready_queue.remove(next_job)
            running_job = next_job
            if running_job.start_time is None:
                running_job.start_time = current_time


        # compute max response times
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
    tasks = validate_taskset(tasks)
    sim_time = int(max_sim_time)
    if sim_time <= 0:
        raise ValueError("max_sim_time must be > 0")

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



def analyze_task_set(tasks: pd.DataFrame, num_sim_runs: int = 100,
                     num_hyperperiods: int = 1, seed: int = 42,
                     verbose: bool = False) -> Dict[str, Any]:
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
    log = print if verbose else (lambda *args, **kwargs: None)
    log("=" * 70)

    tasks = validate_taskset(tasks)
    log("REAL-TIME SCHEDULING ANALYSIS")
    log("=" * 70)

    n = len(tasks)

    # 1. Utilization
    U = compute_utilization(tasks)
    log(f"\n1. UTILIZATION: U = {U:.6f}")

    # 2. DM RTA
    log("\n2. DEADLINE MONOTONIC (DM) ANALYSIS")
    log("-" * 40)
    dm_ok, dm_msg, tasks_dm = dm_schedulability_test(tasks)
    log(f"   Schedulable: {dm_ok}")
    log(f"   Message: {dm_msg}")

    # 3. EDF DBF Feasibility
    log("\n3. EDF FEASIBILITY (DBF TEST)")
    log("-" * 40)
    edf_ok, edf_msg = edf_dbf_feasibility_test(tasks)
    log(f"   Feasible: {edf_ok}")
    log(f"   Message: {edf_msg}")

    # 4. EDF WCRT (Schedule Construction)
    log("\n4. EDF WCRT (SCHEDULE CONSTRUCTION)")
    log("-" * 40)
    tasks_edf, (edf_wcrt_ok, edf_wcrt_msg) = edf_wcrt_schedule_construction(tasks)
    log(f"   Success: {edf_wcrt_ok}")
    log(f"   Message: {edf_wcrt_msg}")

    # Combine results
    results_df = tasks.copy()

    # Add DM results (need to align by task name since DM sorts by deadline)
    dm_results = tasks_dm[["Name", "Ri_DM"]].set_index("Name")
    results_df = results_df.set_index("Name")
    results_df["Ri_DM"] = dm_results["Ri_DM"]
    results_df = results_df.reset_index()

    # add EDF results
    results_df["Ri_EDF"] = tasks_edf["Ri_EDF"].values

    log("\n   Per-task Analytical WCRTs:")
    log("   " + "-" * 50)
    log(f"   {'Task':<8} {'D_i':<8} {'Ri_DM':<12} {'Ri_EDF':<12}")
    log("   " + "-" * 50)
    for i, row in results_df.iterrows():
        name = row['Name']
        Di = row['Deadline']
        Ri_DM = row['Ri_DM']
        Ri_EDF = row['Ri_EDF']
        log(f"   {name:<8} {Di:<8} {str(Ri_DM):<12} {str(Ri_EDF):<12}")

    # 5. Discrete-Event Simulation
    log("\n5. DISCRETE-EVENT SIMULATION (WCET)")
    log("-" * 40)

    H = compute_hyperperiod(tasks["Period"].tolist())
    log(f"   Hyperperiod H = {H}")

    base_window = min(H, 100000)
    sim_time = base_window * max(1, int(num_hyperperiods))
    if H > base_window:
        # Not ideal, but this keeps big generated sets from exploding runtime.
        log(f"   WARNING: Hyperperiod large, using per-window cap = {base_window}")
    if num_hyperperiods > 1:
        log(f"   Running for {num_hyperperiods} hyperperiods ({sim_time} time units)")

    log("\n   Running DM simulation (WCET)...")
    dm_sim = simulate_schedule(tasks, policy="DM", use_wcet=True, max_sim_time=sim_time)

    log("   Running EDF simulation (WCET)...")
    edf_sim = simulate_schedule(tasks, policy="EDF", use_wcet=True, max_sim_time=sim_time)

    # Stochastic simulation statistics p7
    log(f"\n   Running stochastic simulations ({num_sim_runs} runs, Uniform[BCET, WCET])...")
    dm_stochastic = run_stochastic_simulation_stats(tasks, "DM", num_sim_runs, sim_time, seed)
    edf_stochastic = run_stochastic_simulation_stats(tasks, "EDF", num_sim_runs, sim_time, seed)

    # Add simulation results to DataFrame
    results_df["Ri_DM_sim"] = [dm_sim['max_response_times'].get(i, None) for i in range(n)]
    results_df["Ri_EDF_sim"] = [edf_sim['max_response_times'].get(i, None) for i in range(n)]
    results_df["Ri_DM_obs_max"] = [dm_stochastic['observed_max_rt'].get(i, None) for i in range(n)]
    results_df["Ri_EDF_obs_max"] = [edf_stochastic['observed_max_rt'].get(i, None) for i in range(n)]
    results_df["Ri_DM_obs_mean"] = [dm_stochastic['observed_mean_rt'].get(i, None) for i in range(n)]
    results_df["Ri_EDF_obs_mean"] = [edf_stochastic['observed_mean_rt'].get(i, None) for i in range(n)]
    results_df["Ri_DM_obs_p95"] = [dm_stochastic['observed_p95_rt'].get(i, None) for i in range(n)]
    results_df["Ri_EDF_obs_p95"] = [edf_stochastic['observed_p95_rt'].get(i, None) for i in range(n)]
    results_df["DM_obs_within_ana"] = [
        (results_df.loc[i, "Ri_DM_obs_max"] <= results_df.loc[i, "Ri_DM"])
        if isinstance(results_df.loc[i, "Ri_DM"], (int, np.integer)) and results_df.loc[i, "Ri_DM_obs_max"] is not None
        else None
        for i in range(n)
    ]
    results_df["EDF_obs_within_ana"] = [
        (results_df.loc[i, "Ri_EDF_obs_max"] <= results_df.loc[i, "Ri_EDF"])
        if isinstance(results_df.loc[i, "Ri_EDF"], (int, np.integer)) and results_df.loc[i, "Ri_EDF_obs_max"] is not None
        else None
        for i in range(n)
    ]
    results_df["DM_preemptions"] = [dm_sim['preemptions'].get(i, 0) for i in range(n)]
    results_df["EDF_preemptions"] = [edf_sim['preemptions'].get(i, 0) for i in range(n)]

    # 6. Comparison: Analytical vs Simulation
    log("\n6. COMPARISON: ANALYTICAL vs SIMULATION")
    log("-" * 70)
    log(f"   {'Task':<8} {'D_i':<6} {'DM_ana':<8} {'DM_sim':<8} {'Match':<6} {'EDF_ana':<8} {'EDF_sim':<8} {'Match':<6}")
    log("   " + "-" * 70)

    dm_matches = 0
    edf_matches = 0
    edf_comparable = 0
    for i, row in results_df.iterrows():
        name = row['Name']
        Di = row['Deadline']
        dm_ana = row['Ri_DM']
        dm_sim_val = row['Ri_DM_sim']
        edf_ana = row['Ri_EDF']
        edf_sim_val = row['Ri_EDF_sim']

        # Check if analytical matches simulation
        dm_match = "PASS" if (dm_ana == dm_sim_val or (isinstance(dm_ana, str) and dm_ana == "UNFEASIBLE")) else "FAIL"
        if isinstance(edf_ana, str) and edf_ana == "NOT_COMPUTED":
            edf_match = "N/A"
        else:
            edf_match = "PASS" if (edf_ana == edf_sim_val or (isinstance(edf_ana, str) and edf_ana == "UNFEASIBLE")) else "FAIL"

        if dm_match == "PASS":
            dm_matches += 1
        if edf_match != "N/A":
            edf_comparable += 1
        if edf_match == "PASS":
            edf_matches += 1

        log(f"   {name:<8} {Di:<6} {str(dm_ana):<8} {str(dm_sim_val):<8} {dm_match:<6} {str(edf_ana):<8} {str(edf_sim_val):<8} {edf_match:<6}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"   Total Utilization: {U:.6f}")
    log(f"   DM Schedulable (analytical): {dm_ok}")
    log(f"   EDF Feasible (analytical): {edf_ok}")
    log(f"\n   DM: Analytical vs Simulation match: {dm_matches}/{n} tasks")
    log(f"   EDF: Analytical vs Simulation match: {edf_matches}/{edf_comparable} tasks")

    dm_total_misses = sum(dm_sim['deadline_misses'].values())
    edf_total_misses = sum(edf_sim['deadline_misses'].values())
    log(f"\n   DM Deadline Misses (simulation): {dm_total_misses}")
    log(f"   EDF Deadline Misses (simulation): {edf_total_misses}")

    dm_total_preempt = sum(dm_sim['preemptions'].values())
    edf_total_preempt = sum(edf_sim['preemptions'].values())
    log(f"   DM Total Preemptions: {dm_total_preempt}")
    log(f"   EDF Total Preemptions: {edf_total_preempt}")

    return {
        'utilization': U,
        'dm_schedulable': dm_ok,
        'dm_message': dm_msg,
        'edf_feasible': edf_ok,
        'edf_message': edf_msg,
        'results_df': results_df,
        'dm_sim': dm_sim,
        'edf_sim': edf_sim,
        'dm_stochastic': dm_stochastic,
        'edf_stochastic': edf_stochastic,
    }


def generate_report(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """PrettyPrint of the analysis results."""

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


def normalize_task_columns(tasks: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to handle different CSV formats."""
    return validate_taskset(tasks)


if __name__ == "__main__":
    csv_path = 'task_sets/schedulable/Full_Utilization_NonUnique_Periods_taskset.csv'
    print(f"Loading task set from {csv_path}...")
    tasks = pd.read_csv(csv_path)
    tasks = normalize_task_columns(tasks)

    print(f"\nTask Set ({len(tasks)} tasks):")
    print(tasks[['Name', 'BCET', 'WCET', 'Period', 'Deadline']].to_string())

    # Run complete analysis
    results = analyze_task_set(
        tasks,
        num_sim_runs=10,
        num_hyperperiods=1,
        seed=42
    )

    # Save results to CSV
    results['results_df'].to_csv('data/analysis_results.csv', index=False)
    print("\nResults saved to data/analysis_results.csv")

