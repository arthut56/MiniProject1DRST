import pandas as pd

from scheduler_analysis import (
    edf_dbf_feasibility_test,
    normalize_task_columns,
    simulate_schedule,
)


def test_edf_dbf_u_equals_one_with_constrained_deadline_can_be_feasible():
    tasks = pd.DataFrame(
        {
            "Name": ["t1", "t2"],
            "BCET": [1, 1],
            "WCET": [1, 1],
            "Period": [2, 2],
            "Deadline": [1, 2],
        }
    )

    feasible, _ = edf_dbf_feasibility_test(tasks)
    assert feasible is True


def test_simulation_respects_requested_max_sim_time_window():
    tasks = pd.DataFrame(
        {
            "Name": ["t1"],
            "BCET": [1],
            "WCET": [1],
            "Period": [5],
            "Deadline": [5],
        }
    )

    result = simulate_schedule(tasks, policy="DM", use_wcet=True, max_sim_time=15)

    assert result["sim_time"] == 15
    assert len(result["response_times"][0]) == 3


def test_normalize_task_columns_accepts_task_alias_and_missing_bcet():
    raw = pd.DataFrame(
        {
            "Task": ["tau_1"],
            "WCET": [2],
            "Period": [10],
            "Deadline": [10],
        }
    )

    normalized = normalize_task_columns(raw)

    assert "Name" in normalized.columns
    assert "Task" not in normalized.columns
    assert int(normalized.loc[0, "BCET"]) == int(normalized.loc[0, "WCET"])
