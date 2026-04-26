# MiniProject1DRST

Scheduler analysis for DM/EDF: analytical WCRT, schedulability checks, simulation, and comparison utilities.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run Tests

```bash
python3 -m pytest tests/ -v
```

3 regression tests covering edge cases (equal-deadline RTA, EDF boundary, column normalisation).

## Quick Start (recommended)

Runs analysis, batch validation, and plotting (no utilization sweep). Generates all figures needed for the report.

```bash
python3 run_all_experiments.py --quick
```

Outputs:
- `data/analysis_results.csv`
- `data/all_tasksets_results.csv`
- `data/fig8_tc5_rt_samples.csv`
- `data/fig9_arj_u07_u08_u09.csv`
- `data/figures/fig*.png`  ← used by the combined report

## Full Workflow

Includes the utilization sweep (400 samples per U-level, ~2 min) and all plots.

```bash
python3 run_all_experiments.py
```

If `task_sets/generated/sweep/` does not exist, the sweep generates task sets on the fly.

## Individual Components

```bash
# Batch validation of provided task sets
python3 test_all_tasksets.py

# Main analysis on the default task set
python3 scheduler_analysis.py

# Utilization sweep only (writes data/utilization_sweep_results.csv)
python3 experiments.py

# Plot generation only (expects prior CSVs in data/)
python3 visualizations.py
```

`visualizations.py` reads CSV sources for all non-gantt plots:
- Fig4 uses `data/fraction_schedulable_summary.csv`
- Fig5-7 use `data/analysis_results.csv`
- Fig8 uses `data/fig8_tc5_rt_samples.csv`
- Fig9 uses `data/fig9_arj_u07_u08_u09.csv`

Note: `visualizations.py` is read-only with respect to CSV inputs. The full workflow
(`run_all_experiments.py`) generates `fig8/fig9` CSVs explicitly before plotting.

**Fig8 note:** `fig8_tc5_rt_samples.csv` is named after Figure 8 in the report. It uses a
separate constrained-deadline task set (`task_sets/generated/report_fig8_taskset.csv`),
**not** the TC5 task set from Table 5. TC5 has `D_i = T_i`; the Fig8 task set has
`D_i < T_i` for several tasks. See the report caption for clarification.

Figure 8/9 consume task sets in `task_sets/generated/report_fig8_taskset.csv` and
`task_sets/generated/report_fig9_u0*.csv`.

## Layout

- `scheduler_analysis.py` – core DM/EDF analysis + simulation
- `test_all_tasksets.py` – batch validation over `task_sets/`
- `experiments.py` – utilization sweep generation and aggregation
- `visualizations.py` – plot generation to `data/figures/`
- `run_all_experiments.py` – orchestration (quick/full modes)
- `data/` – analysis CSVs and generated figures
- `task_sets/` – provided schedulable/unschedulable sets
- `tests/` – regression test suite
