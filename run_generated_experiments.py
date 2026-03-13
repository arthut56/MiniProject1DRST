#!/usr/bin/env python3
"""Generate task sets and evaluate them with the project schedulability pipeline.

This script bridges `real-time-task-generators/task_generator.py` with
`test_all_tasksets.test_task_set` so generated sets can be analyzed in batch.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from test_all_tasksets import test_task_set


@dataclass
class GenerationConfig:
    utilization: int
    num_tasks: int
    generator_id: int


def parse_csv_ints(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate task sets and evaluate DM/EDF schedulability in batch."
    )
    parser.add_argument("--generator-dir", default="real-time-task-generators")
    parser.add_argument("--output-dir", default="task_sets/generated")
    parser.add_argument("--results-file", default="data/generated_experiment_results.csv")
    parser.add_argument("--utilizations", default="50", help="CSV list, e.g. 20,50,80")
    parser.add_argument("--num-tasks", default="15", help="CSV list, e.g. 10,20")
    parser.add_argument("--generator-ids", default="1", help="CSV list from {0,1,2,3}")
    parser.add_argument("--sets-per-config", type=int, default=1)
    parser.add_argument("--mapping", type=int, default=0)
    parser.add_argument("--npe", type=int, default=4)
    parser.add_argument("--round", action="store_true", dest="round_values")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--glob", default="*.csv", help="Used with --skip-generate")
    return parser


def run_generator(
    generator_dir: Path,
    cfg: GenerationConfig,
    sets_per_config: int,
    mapping: int,
    npe: int,
    round_values: bool,
) -> List[Path]:
    cmd = [
        "python3",
        "task_generator.py",
        "-u",
        str(cfg.utilization),
        "-g",
        str(cfg.generator_id),
        "-n",
        str(cfg.num_tasks),
        "-s",
        str(sets_per_config),
        "-m",
        str(mapping),
        "-p",
        str(npe),
    ]
    if round_values:
        cmd.append("-r")

    subprocess.run(cmd, cwd=str(generator_dir), check=True)

    generated = sorted(generator_dir.glob("taskset-*.csv"))
    if len(generated) < sets_per_config:
        raise RuntimeError(
            f"Expected at least {sets_per_config} generated files, found {len(generated)}"
        )
    return generated[:sets_per_config]


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    tracked_files: List[Path] = []

    if args.skip_generate:
        tracked_files = sorted(output_dir.glob(args.glob))
        if not tracked_files:
            raise RuntimeError(f"No files matched {args.glob} in {output_dir}")
    else:
        generator_dir = Path(args.generator_dir)
        if not generator_dir.exists():
            raise FileNotFoundError(f"Generator directory not found: {generator_dir}")

        util_values = parse_csv_ints(args.utilizations)
        task_values = parse_csv_ints(args.num_tasks)
        generator_ids = parse_csv_ints(args.generator_ids)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for util in util_values:
            for n_tasks in task_values:
                for generator_id in generator_ids:
                    cfg = GenerationConfig(
                        utilization=util,
                        num_tasks=n_tasks,
                        generator_id=generator_id,
                    )

                    generated_paths = run_generator(
                        generator_dir=generator_dir,
                        cfg=cfg,
                        sets_per_config=args.sets_per_config,
                        mapping=args.mapping,
                        npe=args.npe,
                        round_values=args.round_values,
                    )

                    for idx, src in enumerate(generated_paths):
                        dst_name = (
                            f"gen_{timestamp}_u{util}_n{n_tasks}_g{generator_id}_"
                            f"set{idx}.csv"
                        )
                        dst = output_dir / dst_name
                        shutil.move(str(src), str(dst))
                        tracked_files.append(dst)

    for csv_path in tracked_files:
        result = test_task_set(str(csv_path))
        result["source_path"] = str(csv_path)
        all_rows.append(result)

    df = pd.DataFrame(all_rows)
    Path(args.results_file).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.results_file, index=False)

    cols = [
        "file",
        "utilization",
        "num_tasks",
        "dm_schedulable",
        "edf_feasible",
        "dm_sim_misses",
        "edf_sim_misses",
    ]
    present = [c for c in cols if c in df.columns]
    if present:
        print(df[present].to_string(index=False))
    print(f"\nSaved results: {args.results_file}")


if __name__ == "__main__":
    main()


