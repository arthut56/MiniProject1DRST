# MiniProject1DRST

Canonical implementation: `scheduler_analysis.py`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```


## Quick Run

```bash
python3 -m py_compile scheduler_analysis.py test_all_tasksets.py
python3 test_all_tasksets.py
python3 scheduler_analysis.py
```

## Repository Contents

- `scheduler_analysis.py`: analytical and simulation toolchain
- `test_all_tasksets.py`: batch validation on `task_sets/`
- `task_sets/`: provided test scenarios
- `data/`: generated result CSV files
- `SPEC_COMPLIANCE_REVIEW.md`: requirement traceability and compliance notes
- `report/`: LaTeX report sources (`main.tex`, `references.bib`)
- `run_generated_experiments.py`: generator + analysis batch runner

## Generate and Evaluate New Task Sets

```bash
python3 run_generated_experiments.py \
  --generator-dir real-time-task-generators \
  --output-dir task_sets/generated \
  --results-file data/generated_experiment_results.csv \
  --utilizations 20,50,80 \
  --num-tasks 10,20 \
  --generator-ids 1 \
  --sets-per-config 2
```

If task sets are already generated, analyze only existing files:

```bash
python3 run_generated_experiments.py \
  --skip-generate \
  --output-dir task_sets/generated \
  --glob "*.csv" \
  --results-file data/generated_experiment_results.csv
```

## Build the PDF Report

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

