import pandas as pd
import math as m
tasks = pd.read_csv('data/task-set-example.csv')


# 1. Load the tasks
tasks_data = pd.read_csv('data/task-set-example.csv')

def analytical_wcrt_dm(tasks):
    #DEADLINE MONOTONIC
    tasks = tasks.sort_values(by='Deadline').reset_index(drop=True)

    #RATE MONOTONIC
    # tasks = tasks.sort_values(by='Period').reset_index(drop=True)

    # We will store the results in this list then add to the dataframe
    wcrts = []

    for i in range(len(tasks)):
        Ci = tasks.loc[i, 'WCET']
        Di = tasks.loc[i, 'Deadline']

        # Initialize Ri with Ci (the standard starting guess)
        Ri = Ci

        while True:
            interference = 0

            # Calculate interference from all tasks with higher priority (0 to i-1)
            for h in range(0, i):
                Ch = tasks.loc[h, 'WCET']
                Th = tasks.loc[h, 'Period']  # Corrected: Period of the higher-priority task

                # This is the core RTA formula
                interference += m.ceil(Ri / Th) * Ch

            new_Ri = Ci + interference

            # TERMINATION CONDITIONS
            # Condition 1: Converged (Success!)
            if new_Ri == Ri:
                wcrts.append(Ri)
                break

            # Condition 2: Deadline Missed (Unfeasible)
            if new_Ri > Di:
                #wcrts.append(new_Ri)  # Or mark as "FAIL"
                wcrts.append("UNFEASIBLE")  # Or mark as "FAIL"
                break

            # Update Ri for the next iteration of the while loop
            Ri = new_Ri

    # Add the results back to the dataframe
    tasks['Ri'] = wcrts
    return tasks
    #print(tasks[['TaskID', 'WCET', 'Deadline', 'Period', 'Ri']])

def is_schedulable_by_edf(tasks):
    # 1. Basic check: Utilization must be <= 1.0
    U = sum(tasks['WCET'] / tasks['Period'])
    if U > 1.0:
        return False, "Unfeasible: Utilization > 1.0"

    # 2. Determine the check bound (L*)
    # Formula from Page 105/106
    sum_term = sum((tasks['Period'] - tasks['Deadline']) * (tasks['WCET'] / tasks['Period']))
    L_star = sum_term / (1 - U)

    # Also consider the Hyperperiod if L_star is very large
    # For your project, let's collect all absolute deadlines up to L_star
    test_points = []
    for _, task in tasks.iterrows():
        # Generate all deadlines for this task that occur before L_star
        t = task['Deadline']
        while t <= L_star:
            test_points.append(t)
            t += task['Period']

    test_points = sorted(list(set(test_points)))  # Unique, sorted deadlines

    # 3. Check dbf(t) <= t at every test point
    for t in test_points:
        demand = 0
        for _, task in tasks.iterrows():
            # DBF formula
            count = m.floor((t - task['Deadline']) / task['Period']) + 1
            demand += max(0, count) * task['WCET']

        if demand > t:
            return False, f"Failed at t={t}: Demand {demand} > {t}"

    return True, "Schedulable"

print(analytical_wcrt_dm(tasks))
print(is_schedulable_by_edf(tasks))