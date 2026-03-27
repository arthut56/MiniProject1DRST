#!/usr/bin/env python3
"""
Generate Gantt charts for TC1 and TC2.

Outputs:
- fig1_tc1_dm_gantt.png: DM schedule for TC1
- fig2_tc1_edf_gantt.png: EDF schedule for TC1
- fig3_tc2_comparison_gantt.png: DM vs EDF for TC2
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from scheduler_analysis import compute_hyperperiod

TC1_TASKS = {
    'Name': ['tau_1', 'tau_2', 'tau_3'],
    'BCET': [1, 2, 1],
    'WCET': [1, 2, 1],
    'Period': [4, 6, 8],
    'Deadline': [4, 6, 8],
}

TC2_TASKS = {
    'Name': ['tau_1', 'tau_2'],
    'BCET': [2, 4],
    'WCET': [2, 4],
    'Period': [5, 7],
    'Deadline': [5, 7],
}

def simulate_schedule_gantt(tasks, policy='DM', time_limit=None):
    """
    Simulate a preemptive single-CPU schedule for Gantt chart rendering.
    
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

    jobs.sort(key=lambda j: (j.release_time, j.task_id, j.job_id))
    
    current_time = 0
    ready_queue = []
    job_index = 0
    running_job = None
    events = []  # (task_id, start, end)

    def push_ready(job):
        if policy == 'DM':
            key = (job.priority, job.task_id, job.job_id)
        else:
            key = (job.absolute_deadline, job.task_id, job.job_id)
        heapq.heappush(ready_queue, (key, job))

    def pop_ready():
        return heapq.heappop(ready_queue)[1]
    
    while job_index < len(jobs) or ready_queue or running_job:
        # release jobs at current time
        while job_index < len(jobs) and jobs[job_index].release_time == current_time:
            push_ready(jobs[job_index])
            job_index += 1

        # dispatch if idle
        if running_job is None and ready_queue:
            running_job = pop_ready()

        next_release = jobs[job_index].release_time if job_index < len(jobs) else float('inf')
        next_completion = (
            current_time + running_job.remaining_time
            if running_job is not None else float('inf')
        )
        next_event_time = min(next_release, next_completion, time_limit)

        if next_event_time == float('inf') or next_event_time == current_time:
            if running_job is None:
                break
            next_event_time = min(current_time + running_job.remaining_time, time_limit)

        # execute running job segment
        if running_job is not None and next_event_time > current_time:
            running_job.remaining_time -= (next_event_time - current_time)
            events.append((running_job.task_id, current_time, next_event_time))

        current_time = int(next_event_time)

        # handle completion
        if running_job is not None and running_job.remaining_time == 0:
            running_job = None

        # release and check preemption at the same timestamp
        while job_index < len(jobs) and jobs[job_index].release_time == current_time:
            push_ready(jobs[job_index])
            job_index += 1

        if running_job is not None and ready_queue:
            candidate = ready_queue[0][1]
            candidate_key = ready_queue[0][0]
            running_key = (
                (running_job.priority, running_job.task_id, running_job.job_id)
                if policy == 'DM'
                else (running_job.absolute_deadline, running_job.task_id, running_job.job_id)
            )
            if candidate_key < running_key:
                push_ready(running_job)
                running_job = pop_ready()

        if current_time >= time_limit:
            break

    merged = []
    for task_id, start, end in events:
        if end <= start:
            continue
        if merged and merged[-1][0] == task_id and merged[-1][2] == start:
            merged[-1] = (task_id, merged[-1][1], end)
        else:
            merged.append((task_id, start, end))

    return merged

def draw_gantt(ax, events, tasks, title, time_limit, wcrts=None, deadline_misses=None, show_all_deadlines=True):
    """Draw Gantt chart."""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    y_pos = {}
    for i, row in tasks.iterrows():
        y_pos[i] = i
    
    for task_id, start, end in events:
        ax.barh(y_pos[task_id], end - start, left=start, 
               height=0.6, color=colors[task_id % len(colors)],
               edgecolor='black', linewidth=1.5)

    if show_all_deadlines:
        for task_id, row in tasks.iterrows():
            T = int(row['Period'])
            D = int(row['Deadline'])
            k = 0
            while True:
                abs_deadline = k * T + D
                if abs_deadline > time_limit:
                    break
                ax.vlines(
                    abs_deadline,
                    y_pos[task_id] - 0.32,
                    y_pos[task_id] + 0.32,
                    colors='dimgray',
                    linestyles=':',
                    linewidth=1.0,
                    alpha=0.9,
                )
                k += 1
    
    if wcrts:
        for i, wcrt_val in enumerate(wcrts):
            if wcrt_val is not None:
                if isinstance(wcrt_val, tuple):
                    x_pos, label = wcrt_val
                else:
                    x_pos, label = wcrt_val, str(wcrt_val)
                ax.annotate(f'WCRT={label}', xy=(x_pos, i), xytext=(x_pos + 0.5, i + 0.2),
                            arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=0),
                            fontsize=9, fontweight='bold', color='black')
                ax.axvline(x=x_pos, ymin=(i-0.3)/len(tasks), ymax=(i+0.3)/len(tasks), 
                           color='red', linestyle='--', linewidth=1.5)
                           
    if deadline_misses:
        for miss_time, i in deadline_misses:
            ax.annotate('Miss!', xy=(miss_time, i), xytext=(miss_time + 1, i - 0.2),
                        arrowprops=dict(facecolor='red', arrowstyle='->', shrinkA=0, shrinkB=0),
                        fontsize=10, fontweight='bold', color='red')
            ax.axvline(x=miss_time, color='red', linestyle='-', linewidth=2, alpha=0.7)

    task_names = [row['Name'] for _, row in tasks.iterrows()]
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(task_names)
    
    ax.set_xlim(0, time_limit)
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Task', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

def main():
    os.makedirs('data/figures', exist_ok=True)

    tc1_df = pd.DataFrame(TC1_TASKS)
    tc2_df = pd.DataFrame(TC2_TASKS)

    H_tc1 = compute_hyperperiod(tc1_df['Period'].tolist())
    H_tc2 = compute_hyperperiod(tc2_df['Period'].tolist())

    tc1_dm_events = simulate_schedule_gantt(tc1_df, policy='DM', time_limit=H_tc1)

    tc1_edf_events = simulate_schedule_gantt(tc1_df, policy='EDF', time_limit=H_tc1)

    tc2_time_limit = min(H_tc2, 21)
    tc2_dm_events = simulate_schedule_gantt(tc2_df, policy='DM', time_limit=tc2_time_limit)
    tc2_edf_events = simulate_schedule_gantt(tc2_df, policy='EDF', time_limit=tc2_time_limit)

    tc1_desc = "tau_1(C=1,T=4,D=4), tau_2(C=2,T=6,D=6), tau_3(C=1,T=8,D=8)"
    tc2_desc = "tau_1(C=2,T=5,D=5), tau_2(C=4,T=7,D=7)"

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    draw_gantt(
        ax1,
        tc1_dm_events,
        tc1_df,
        "TC1: DM schedule (U=0.708, H=24)\n" + tc1_desc,
        H_tc1,
        wcrts=[1, 3, 4],
    )
    fig1.tight_layout()
    fig1.savefig('data/figures/fig1_tc1_dm_gantt.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    draw_gantt(
        ax2,
        tc1_edf_events,
        tc1_df,
        "TC1: EDF schedule (U=0.708, H=24)\n" + tc1_desc,
        H_tc1,
        wcrts=[1, 3, 4],
    )
    fig2.tight_layout()
    fig2.savefig('data/figures/fig2_tc1_edf_gantt.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6))
    draw_gantt(
        ax3a,
        tc2_dm_events,
        tc2_df,
        "TC2: DM schedule (U=0.971) — infeasible\n" + tc2_desc,
        tc2_time_limit,
        deadline_misses=[(7, 1)],
    )
    draw_gantt(
        ax3b,
        tc2_edf_events,
        tc2_df,
        "TC2: EDF schedule (U=0.971) — feasible, WCRTs=(4,6)\n" + tc2_desc,
        tc2_time_limit,
        wcrts=[(14, 4), (6, 6)],
    )
    fig3.tight_layout()
    fig3.savefig('data/figures/fig3_tc2_comparison_gantt.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)



if __name__ == "__main__":
    main()
