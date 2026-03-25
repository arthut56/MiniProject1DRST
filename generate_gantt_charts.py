#!/usr/bin/env python3
"""
Generate Gantt charts for TC1 and TC2 as per report specification.

Outputs:
- fig1_tc1_dm_gantt.png: DM schedule for TC1
- fig2_tc1_edf_gantt.png: EDF schedule for TC1
- fig3_tc2_comparison_gantt.png: DM vs EDF for TC2
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from scheduler_analysis import (
    dm_schedulability_test, edf_wcrt_schedule_construction,
    compute_hyperperiod, normalize_task_columns
)
import os

# Create task sets for TC1 and TC2
TC1_TASKS = {
    'Name': ['tau_1', 'tau_2', 'tau_3'],
    'BCET': [1, 2, 1],
    'WCET': [1, 2, 1],
    'Period': [4, 6, 8],
    'Deadline': [4, 6, 8]
}

TC2_TASKS = {
    'Name': ['tau_1', 'tau_2'],
    'BCET': [2, 4],
    'WCET': [2, 4],
    'Period': [5, 7],
    'Deadline': [5, 7]
}

def simulate_schedule_gantt(tasks, policy='DM', time_limit=None):
    """
    Simulate schedule and return (time, task_id) pairs for Gantt chart.
    
    Returns:
        List of (task_id, start_time, end_time) tuples
    """
    import heapq
    from dataclasses import dataclass
    
    @dataclass(order=True)
    class Job:
        priority: int
        task_id: int
        job_id: int
        release_time: int
        absolute_deadline: int
        execution_time: int
        remaining_time: int
    
    if time_limit is None:
        time_limit = compute_hyperperiod(tasks['Period'].tolist())
    
    # Generate all jobs
    jobs = []
    for i, row in tasks.iterrows():
        T = int(row['Period'])
        D = int(row['Deadline'])
        C = int(row['WCET'])
        k = 0
        while k * T < time_limit:
            r = k * T
            d = r + D
            if policy == 'DM':
                priority = int(row['Deadline'])
            else:  # EDF
                priority = d
            
            job = Job(
                priority=priority,
                task_id=i,
                job_id=k,
                release_time=r,
                absolute_deadline=d,
                execution_time=C,
                remaining_time=C
            )
            jobs.append(job)
            k += 1
    
    # Event-driven simulation
    current_time = 0
    ready_queue = []
    job_index = 0
    running_job = None
    events = []  # (task_id, start, end)
    completed = set()
    
    while job_index < len(jobs) or ready_queue or running_job:
        # Release jobs
        while job_index < len(jobs) and jobs[job_index].release_time == current_time:
            heapq.heappush(ready_queue, jobs[job_index])
            job_index += 1
        
        # If no job running, pick from queue
        if running_job is None and ready_queue:
            running_job = heapq.heappop(ready_queue)
        
        # Find next event
        if running_job:
            next_event_time = running_job.release_time + running_job.execution_time
            if job_index < len(jobs):
                next_event_time = min(next_event_time, jobs[job_index].release_time)
        elif job_index < len(jobs):
            next_event_time = jobs[job_index].release_time
        else:
            break
        
        # Execute running job
        if running_job:
            start = current_time
            end = next_event_time
            events.append((running_job.task_id, start, end))
            running_job = None
        
        current_time = next_event_time
    
    return events

def draw_gantt(ax, events, tasks, title, time_limit):
    """Draw Gantt chart."""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    y_pos = {}
    for i, row in tasks.iterrows():
        y_pos[i] = i
    
    for task_id, start, end in events:
        ax.barh(y_pos[task_id], end - start, left=start, 
               height=0.6, color=colors[task_id % len(colors)],
               edgecolor='black', linewidth=1.5)
    
    # Add task names on y-axis
    task_names = [row['Name'] for _, row in tasks.iterrows()]
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    
    ax.set_xlim(0, time_limit)
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Task', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

# Generate TC1 Gantt charts
print("Generating Gantt charts for TC1 and TC2...")

tc1_df = pd.DataFrame(TC1_TASKS)
tc2_df = pd.DataFrame(TC2_TASKS)

H_tc1 = compute_hyperperiod(tc1_df['Period'].tolist())
H_tc2 = compute_hyperperiod(tc2_df['Period'].tolist())

# TC1 DM
tc1_dm_events = simulate_schedule_gantt(tc1_df, policy='DM', time_limit=H_tc1)

# TC1 EDF
tc1_edf_events = simulate_schedule_gantt(tc1_df, policy='EDF', time_limit=H_tc1)

# TC2 DM and EDF
tc2_dm_events = simulate_schedule_gantt(tc2_df, policy='DM', time_limit=min(H_tc2, 35))
tc2_edf_events = simulate_schedule_gantt(tc2_df, policy='EDF', time_limit=min(H_tc2, 35))

# Create figures
fig1, ax1 = plt.subplots(figsize=(10, 4))
draw_gantt(ax1, tc1_dm_events, tc1_df, 'TC1: DM Schedule (U=0.708)', H_tc1)
fig1.tight_layout()
fig1.savefig('data/figures/fig1_tc1_dm_gantt.png', dpi=150, bbox_inches='tight')
print("✓ fig1_tc1_dm_gantt.png")
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4))
draw_gantt(ax2, tc1_edf_events, tc1_df, 'TC1: EDF Schedule (U=0.708)', H_tc1)
fig2.tight_layout()
fig2.savefig('data/figures/fig2_tc1_edf_gantt.png', dpi=150, bbox_inches='tight')
print("✓ fig2_tc1_edf_gantt.png")
plt.close(fig2)

# TC2 Comparison (DM vs EDF)
fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6))
draw_gantt(ax3a, tc2_dm_events, tc2_df, 'TC2: DM Schedule (U=0.971) - Infeasible', min(H_tc2, 35))
draw_gantt(ax3b, tc2_edf_events, tc2_df, 'TC2: EDF Schedule (U=0.971) - Feasible', min(H_tc2, 35))
fig3.tight_layout()
fig3.savefig('data/figures/fig3_tc2_comparison_gantt.png', dpi=150, bbox_inches='tight')
print("✓ fig3_tc2_comparison_gantt.png")
plt.close(fig3)

print("\nGantt charts generated successfully!")
print("Output directory: data/figures/")

