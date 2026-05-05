# MiniProject1DRST

Scheduler analysis for periodic real-time tasks under Deadline Monotonic (DM) and Earliest Deadline First (EDF) scheduling.

Computes exact DM WCRTs via response-time analysis, exact EDF WCRTs via hyperperiod simulation, and EDF schedulability via the Processor Demand Criterion. A discrete-event simulator cross-checks analytical bounds under both deterministic and stochastic execution.

---

## Group 21
    Arturo Cortes - s225187
    Kasim Hussain - s225165
    Landon Hassin - s252773
    Matthew Asano - s225134

---

## Features

| Feature | Description |
|---------|-------------|
| DM analytical WCRT | Exact response-time analysis (RTA) with correct tie-breaking for equal deadlines |
| EDF analytical WCRT | Exact WCRT via deterministic hyperperiod simulation at WCET |
| EDF schedulability | Processor Demand Criterion (PDC) check for constrained deadlines |
| DM vs EDF comparison | Per-task WCRT comparison, schedulability rates over utilization sweep |
| Discrete-event simulator | Deterministic (WCET) and stochastic (truncated-normal) modes; cross-checks analytical bounds |
| Preemption analysis | Total preemption counts under DM and EDF across all test cases |
| Jitter analysis | Absolute response jitter (ARJ) per task at U=0.7, 0.8, 0.9 |
| Overload analysis | Deadline miss distribution under DM and EDF at U>1 |
| Utilization sweep | Fraction of schedulable task sets from U=0.5 to U=1.0 (400 sets per level) |

---

## Setup

```bash
python3 -m pip install -r requirements.txt
```

---

## Run Tests

```bash
python3 -m pytest tests/ -v
```

3 regression tests covering edge cases (equal-deadline RTA, EDF boundary, column normalisation).

---

## Quick Start

Runs analysis, batch validation, and plotting (no utilization sweep). Generates all figures needed for the report.

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
CLAUDE.md
Includes the utilization sweep (400 samples per U-level, ~2-5 min) and all plots.

```bash
python3 run_all_experiments.py
```

---

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

---

## Individual Scripts

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

---

## Output Files

All non-figure outputs written to `data/`:

| File | Contents |
|------|----------|
| `analysis_results.csv` | Per-task DM/EDF WCRTs, schedulability flags, preemption counts |
| `all_tasksets_results.csv` | Batch validation results across all provided task sets |
| `utilization_sweep_results.csv` | Per-U-level schedulability fractions (full workflow only) |
| `fraction_schedulable_summary.csv` | Aggregated schedulability rates per utilization level |
| `fig8_tc5_rt_samples.csv` | Stochastic response-time samples for the constrained-deadline task set |
| `fig9_arj_u07_u08_u09.csv` | ARJ per task at U=0.7, 0.8, 0.9 under DM and EDF |
| `figures/fig*.png` | All generated plots |

---

## Project Structure

```
MiniProject1DRST/
├── scheduler_analysis.py     # Core DM/EDF analysis + simulation
├── test_all_tasksets.py      # Batch validation over task_sets/
├── experiments.py            # Utilization sweep generation and aggregation
├── visualizations.py         # Plot generation to data/figures/
├── run_all_experiments.py    # Orchestration (quick/full modes)
├── app.py                    # Flask web GUI
├── templates/                # HTML templates for web GUI
├── static/                   # Static assets for web GUI
├── generate_gantt_charts.py  # Standalone Gantt chart generation
├── uunifast.py               # UUniFast task set generation
├── data/                     # Analysis CSVs and generated figures
├── task_sets/                # Provided schedulable/unschedulable sets
├── tests/                    # Regression test suite
└── README.md
```
