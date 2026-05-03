# MiniProject1DRST

Scheduler analysis for DM/EDF: analytical WCRT, schedulability checks, simulation, and comparison utilities.

---

## Group 21
    Arturo Cortes - s225187
    Kasim Hussain - s225165
    Landon Hassin - s252773
    Matthew Asano - s225134

---

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Web GUI

```bash
python3 app.py
```

Open `http://localhost:5000`.

| Page | What it does |
|------|-------------|
| Dashboard | Stats overview, quick links to bundled task sets |
| Analyze Task Set | Upload CSV, enter tasks manually, or pick a bundled set: runs DM + EDF analysis and shows inline plots |
| Gantt Charts | Generate TC1/TC2 built-in charts or custom task set charts |
| Utilization Sweep | Configure and run `experiments.run_utilization_sweep`: shows fraction-schedulable table + plot |
| Overload Analysis | Run `experiments.run_overload_deadline_miss_analysis`: shows miss fraction by priority rank |
| Figures | Gallery of all PNGs in `data/figures/` with download |
| Data Files | Browse and download any CSV in `data/` |

CLI and all existing scripts remain unchanged. The GUI sits on top and does not modify any of them.

## Run Tests

```bash
python3 -m pytest tests/ -v
```

3 regression tests covering edge cases (equal-deadline RTA, EDF boundary, column normalisation).

## Quick Start

Runs analysis, batch validation, and plotting (skips utilization sweep, since it can take a while to run).

```bash
python3 run_all_experiments.py --quick
```

Outputs:
- `data/analysis_results.csv`
- `data/all_tasksets_results.csv`
- `data/fig8_tc5_rt_samples.csv`
- `data/fig9_arj_u07_u08_u09.csv`
- `data/figures/fig*.png`

## Full Workflow

Includes the utilization sweep (400 samples per U-level, ~2-5 min) and all plots.

```bash
python3 run_all_experiments.py
```

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
- Fig8 uses `data/fig8_tc5_rt_samples.csv`*
- Fig9 uses `data/fig9_arj_u07_u08_u09.csv`*

Note: `visualizations.py` is read-only with respect to CSV inputs. The full workflow
(`run_all_experiments.py`) generates `fig8/fig9` CSVs explicitly before plotting.

Note on Fig8: `fig8_tc5_rt_samples.csv` uses a separate constrained-deadline task set
(`task_sets/generated/report_fig8_taskset.csv`), not the TC5 task set from Table 5.
TC5 has `D_i = T_i`; the Fig8 task set has `D_i < T_i` for several tasks. See the
report caption for clarification.

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
