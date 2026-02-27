import math as m
from math import lcm
import heapq
import pandas as pd

def hyperperiod(periods):
    periods = [int(x) for x in periods]
    H = 1
    for p in periods:
        H = lcm(H, p)
    return H

def dbf(tasks: pd.DataFrame, t: int) -> int:
    """
    Demand-bound function for synchronous periodic tasks with D_i <= T_i:
      dbf(t) = sum_i floor((t + T_i - D_i)/T_i) * C_i
    (Eq. 4.38 in the chapter)
    """
    demand = 0
    for _, task in tasks.iterrows():
        C = int(task["WCET"])
        T = int(task["Period"])
        D = int(task["Deadline"])
        demand += (m.floor((t + T - D) / T) * C)
    return demand

def is_schedulable_by_edf_ch46(tasks: pd.DataFrame):
    """
    Chapter 4.6-style EDF feasibility test:
      - If U > 1 => infeasible
      - Test only at absolute deadlines
      - Only need to test up to min(H, L*)
    Assumes synchronous release (offsets = 0) and constrained deadlines (D <= T).
    """
    tasks = tasks.copy().reset_index(drop=True)

    # Basic utilization check
    U = float((tasks["WCET"] / tasks["Period"]).sum())
    if U > 1.0:
        return False, f"Unfeasible: U={U:.6f} > 1"

    # Hyperperiod bound (test only up to H)
    H = hyperperiod(tasks["Period"].tolist())

    # L* bound (Eq. shown in Sec. 4.6.2)
    # L* = sum_i (T_i - D_i) U_i / (1 - U)
    if abs(1.0 - U) < 1e-12:
        L_star = H  # U==1 => just fall back to H
    else:
        sum_term = float(((tasks["Period"] - tasks["Deadline"]) * (tasks["WCET"] / tasks["Period"])).sum())
        L_star = sum_term / (1.0 - U)

    L_max = min(H, int(m.floor(L_star))) if L_star > 0 else 0

    # Collect absolute deadlines up to L_max (only those matter)
    test_points = set()
    for _, task in tasks.iterrows():
        T = int(task["Period"])
        D = int(task["Deadline"])
        # deadlines: D + kT
        k = 0
        while True:
            d = D + k * T
            if d <= 0:
                k += 1
                continue
            if d > L_max:
                break
            test_points.add(d)
            k += 1

    # If L_max=0, nothing to test beyond utilization
    for t in sorted(test_points):
        if dbf(tasks, t) > t:
            return False, f"Failed at t={t}: dbf(t)={dbf(tasks, t)} > {t}"

    return True, "Schedulable"

def wcrt_edf_analytical(tasks: pd.DataFrame):
    """
    Analytical WCRT computation for EDF using busy-period analysis.

    For EDF with constrained deadlines (D <= T), we compute the worst-case
    response time using the synchronous busy period approach.

    The WCRT for task i is found by iterating:
        L^(k+1) = sum_j ceil(L^(k) / T_j) * C_j
    until convergence, then checking jobs of task i within this busy period.
    """
    tasks = tasks.copy().reset_index(drop=True)

    # First check schedulability
    ok, msg = is_schedulable_by_edf_ch46(tasks)
    if not ok:
        tasks["Ri_EDF"] = ["UNFEASIBLE"] * len(tasks)
        return tasks, (ok, msg)

    n = len(tasks)

    # Compute the synchronous busy period L
    # L = sum of all interference when all tasks release at time 0
    total_C = int(tasks["WCET"].sum())
    L = total_C  # Initial guess

    max_iterations = 10000
    for _ in range(max_iterations):
        new_L = 0
        for j in range(n):
            Cj = int(tasks.loc[j, "WCET"])
            Tj = int(tasks.loc[j, "Period"])
            new_L += m.ceil(L / Tj) * Cj

        if new_L == L:
            break
        if new_L > 10**9:  # Safety limit
            L = new_L
            break
        L = new_L

    # For each task, find WCRT by checking all jobs in the busy period
    wcrts = []

    for i in range(n):
        Ci = int(tasks.loc[i, "WCET"])
        Ti = int(tasks.loc[i, "Period"])
        Di = int(tasks.loc[i, "Deadline"])

        max_R = 0

        # Check each job of task i in the busy period [0, L]
        num_jobs = m.ceil(L / Ti)

        for k in range(num_jobs):
            # Job k of task i: released at r = k*Ti, deadline d = k*Ti + Di
            r = k * Ti
            d = r + Di

            # Find completion time using iterative workload calculation
            # W^(m+1)(t) = (k+1)*Ci + sum_{j!=i} ceil((t)/Tj) * Cj
            #              + sum of jobs of other tasks with deadline <= d

            # For EDF: job completes when workload of higher-priority jobs is done
            # Higher priority = earlier absolute deadline

            w = Ci  # Start with own execution

            for iteration in range(max_iterations):
                new_w = (k + 1) * Ci  # Jobs 0..k of task i

                for j in range(n):
                    if j == i:
                        continue
                    Cj = int(tasks.loc[j, "WCET"])
                    Tj = int(tasks.loc[j, "Period"])
                    Dj = int(tasks.loc[j, "Deadline"])

                    # Count jobs of task j that can interfere
                    # A job of j interferes if its deadline <= completion time of job (i,k)
                    # and it's released before completion
                    # Jobs of j: released at m*Tj, deadline m*Tj + Dj
                    # Interferes if m*Tj + Dj <= d (deadline of job (i,k))
                    #            and m*Tj < w (released before completion)

                    # Number of jobs with deadline <= d
                    if Dj <= d:
                        jobs_with_earlier_deadline = m.floor((d - Dj) / Tj) + 1
                    else:
                        jobs_with_earlier_deadline = 0

                    # Number of jobs released before w
                    jobs_released = m.ceil(w / Tj)

                    # Interference is minimum of both constraints
                    interfering_jobs = min(jobs_with_earlier_deadline, jobs_released)
                    new_w += max(0, interfering_jobs) * Cj

                if new_w == w:
                    break
                if new_w > d:
                    # Deadline miss
                    w = new_w
                    break
                w = new_w

            # Response time for this job
            R = w - r
            if R > Di:
                tasks["Ri_EDF"] = ["UNFEASIBLE"] * n
                return tasks, (False, f"Task {i} job {k} misses deadline: R={R} > D={Di}")

            max_R = max(max_R, R)

        wcrts.append(max_R)

    tasks["Ri_EDF"] = wcrts
    return tasks, (True, "WCRT computed analytically using busy-period analysis")

# --- Usage (replaces your EDF parts) ---
tasks = pd.read_csv('data/task-set-example.csv')

ok, msg = is_schedulable_by_edf_ch46(tasks)
print(ok, msg)

tasks_wcrt, info = wcrt_edf_analytical(tasks)
print(info)
print(tasks_wcrt)