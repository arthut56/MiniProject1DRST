# MiniProject1DRST

Scheduler analysis for DM/EDF: analytical WCRT, schedulability checks, simulation, and comparison utilities.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Quick Start (recommended)

Runs analysis, batch validation, and plotting (no utilization sweep).

```bash
python3 run_all_experiments.py --quick
```

Outputs:
- `data/analysis_results.csv`
- `data/all_tasksets_results.csv`
- `data/figures/fig*.png`

## Full Workflow

Includes the utilization sweep (large: 500 samples per U-level) and all plots.

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

## Layout

- `scheduler_analysis.py` – core DM/EDF analysis + simulation
- `test_all_tasksets.py` – batch validation over `task_sets/`
- `experiments.py` – utilization sweep generation and aggregation
- `visualizations.py` – plot generation to `data/figures/`
- `run_all_experiments.py` – orchestration (quick/full modes)
- `data/` – analysis CSVs and generated figures
- `task_sets/` – provided schedulable/unschedulable sets
