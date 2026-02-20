import pandas as pd
import math as m
tasks = pd.read_csv('data/task-set-example.csv')


for i in range(0,len(tasks)):
    Ci = tasks[tasks["TaskID"] == i]['WCET'].item()
    sum = 0
    previousR = Ci
    for h in range(1, i):
        Ch = tasks[tasks["TaskID"] == h]['WCET'].item()
        Th = tasks[tasks["TaskID"] == i]['Period'].item()
        print(Ch, Th)
        sum += m.ceil(previousR/Th)*Ch

    Ri = Ci + sum
    tasks.loc[tasks["TaskID"] == i, "Ri"] = Ri

print(tasks)

#
# # 1. Load the tasks
# tasks = pd.read_csv('data/task-set-example.csv')
#
# # 2. IMPORTANT: Sort tasks by priority!
# # For Deadline Monotonic, tasks with the smallest Deadline have the highest priority.
# tasks = tasks.sort_values(by='Deadline').reset_index(drop=True)

# # We will store the results in this list then add to the dataframe
# wcrts = []
#
# for i in range(len(tasks)):
#     Ci = tasks.loc[i, 'WCET']
#     Di = tasks.loc[i, 'Deadline']
#
#     # Initialize Ri with Ci (the standard starting guess)
#     Ri = Ci
#
#     while True:
#         interference = 0
#
#         # Calculate interference from all tasks with higher priority (0 to i-1)
#         for h in range(0, i):
#             Ch = tasks.loc[h, 'WCET']
#             Th = tasks.loc[h, 'Period']  # Corrected: Period of the higher-priority task
#
#             # This is the core RTA formula
#             interference += m.ceil(Ri / Th) * Ch
#
#         new_Ri = Ci + interference
#
#         # TERMINATION CONDITIONS
#         # Condition 1: Converged (Success!)
#         if new_Ri == Ri:
#             wcrts.append(Ri)
#             break
#
#         # Condition 2: Deadline Missed (Unfeasible)
#         if new_Ri > Di:
#             wcrts.append(new_Ri)  # Or mark as "FAIL"
#             break
#
#         # Update Ri for the next iteration of the while loop
#         Ri = new_Ri
#
# # Add the results back to the dataframe
# tasks['Ri'] = wcrts
# print(tasks[['TaskID', 'WCET', 'Deadline', 'Period', 'Ri']])