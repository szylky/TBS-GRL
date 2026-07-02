from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_BASE_DIR = Path(__file__).resolve().parent

PRESETS = {
    "DMEPOS": {
        "project_dir": "SZYLKY_DMEPOS",
        "dataset_prefix": "DMEPOS",
        "required_files": ["DMEPOS_Train.csv", "DMEPOS_Test.csv"],
    },
    "Part_B": {
        "project_dir": "SZYLKY_Part_B",
        "dataset_prefix": "Part_B",
        "required_files": ["Part_B_Train.csv", "Part_B_Test.csv"],
    },
    "Part_D": {
        "project_dir": "SZYLKY_Part_D",
        "dataset_prefix": "Part_D",
        "required_files": ["Part_D_Train.csv", "Part_D_Test.csv"],
    },
}

RESULT_COLUMNS = [
    "数据集编号",
    "PR-AUC",
    "ROC-AUC",
    "Precision@1%",
    "Recall@1%",
    "F1@1%",
    "Precision@5%",
    "Recall@5%",
    "F1@5%",
    "Lift@5%",
    "总运行耗时",
]

METRIC_MAP = {
    "PR-AUC": "pr_auc",
    "ROC-AUC": "roc_auc",
    "Precision@1%": "precision_at_1pct",
    "Recall@1%": "recall_at_1pct",
    "F1@1%": "f1_at_1pct",
    "Precision@5%": "precision_at_5pct",
    "Recall@5%": "recall_at_5pct",
    "F1@5%": "f1_at_5pct",
    "Lift@5%": "lift_at_5pct",
}


def build_parser(default_project: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Train.py/Test.py on 5 fold datasets and summarize metrics."
    )
    parser.add_argument(
        "--project",
        choices=sorted(PRESETS),
        default=default_project,
        required=default_project is None,
        help="Project preset.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional explicit project root. Overrides the preset project path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Summary CSV path. Defaults to <project>_5datasets_results.csv.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory for per-dataset Train.py/Test.py logs. Defaults to <project>_5datasets_logs.",
    )
    parser.add_argument("--start", type=int, default=0, help="First dataset index, inclusive.")
    parser.add_argument("--end", type=int, default=4, help="Last dataset index, inclusive.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run Train.py and Test.py.",
    )
    return parser


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_progress(done: int, total: int, dataset_name: str, stage: str) -> None:
    width = 20
    filled = int(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {done}/{total} {dataset_name} {stage}", flush=True)


def patch_dataset_dir(config_path: Path, dataset_dir: str) -> str:
    original = config_path.read_text(encoding="utf-8")
    patched, count = re.subn(
        r'dataset_dir:\s*str\s*=\s*"[^"]+"',
        f'dataset_dir: str = "{dataset_dir}"',
        original,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"Could not find dataset_dir in {config_path}")
    config_path.write_text(patched, encoding="utf-8")
    return original


def restore_config(config_path: Path, original_text: str) -> None:
    config_path.write_text(original_text, encoding="utf-8")


def run_step(command: list[str], cwd: Path, log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}. See log: {log_path}")


def read_metrics(metrics_path: Path) -> dict[str, str]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Metrics file is empty: {metrics_path}")
    return rows[0]


def write_summary(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate_project(project_root: Path, preset: dict[str, object], dataset_indices: range) -> None:
    required = [
        project_root / "Train.py",
        project_root / "Test.py",
        project_root / "00_Global_Config" / "config.py",
        project_root / "01_Dataset",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files/folders:\n" + "\n".join(missing))

    dataset_prefix = str(preset["dataset_prefix"])
    required_files = [str(item) for item in preset.get("required_files", [])]
    required_dirs = [str(item) for item in preset.get("required_dirs", [])]
    for idx in dataset_indices:
        dataset_dir = project_root / "01_Dataset" / f"{dataset_prefix}_{idx}"
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
        for filename in required_files:
            if not (dataset_dir / filename).exists():
                raise FileNotFoundError(f"{dataset_dir} is missing {filename}")
        for dirname in required_dirs:
            if not (dataset_dir / dirname).is_dir():
                raise FileNotFoundError(f"{dataset_dir} is missing directory {dirname}")


def run_batch(args: argparse.Namespace) -> None:
    preset = PRESETS[args.project]
    project_root = (args.project_root or (PROJECT_BASE_DIR / str(preset["project_dir"]))).resolve()
    output_csv = args.output_csv or Path(f"{args.project}_5datasets_results.csv")
    logs_dir = args.logs_dir or Path(f"{args.project}_5datasets_logs")
    dataset_indices = range(args.start, args.end + 1)
    total = len(list(dataset_indices))
    if total <= 0:
        raise ValueError("--end must be greater than or equal to --start")

    validate_project(project_root, preset, dataset_indices)
    logs_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_root / "00_Global_Config" / "config.py"
    original_config = config_path.read_text(encoding="utf-8")
    batch_id = datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_prefix = str(preset["dataset_prefix"])
    summary_rows: list[dict[str, str]] = []
    all_start = time.perf_counter()

    print(f"Project: {project_root}", flush=True)
    print(f"Summary CSV: {output_csv.resolve()}", flush=True)
    print(f"Logs: {logs_dir.resolve()}", flush=True)

    try:
        for position, idx in enumerate(dataset_indices, start=1):
            dataset_name = f"{dataset_prefix}_{idx}"
            run_id = f"batch_{batch_id}_{dataset_name}"
            dataset_start = time.perf_counter()

            print_progress(position - 1, total, dataset_name, "start")
            restore_config(config_path, original_config)
            patch_dataset_dir(config_path, f"01_Dataset/{dataset_name}")

            train_log = logs_dir / f"{dataset_name}_Train.log"
            test_log = logs_dir / f"{dataset_name}_Test.log"

            print_progress(position - 1, total, dataset_name, "Train.py")
            run_step([args.python, "Train.py", "--run-id", run_id], project_root, train_log)

            print_progress(position - 1, total, dataset_name, "Test.py")
            run_step([args.python, "Test.py", "--run-id", run_id], project_root, test_log)

            metrics_path = (
                project_root
                / "05_Outputs"
                / run_id
                / "04_Group_Classification"
                / "test"
                / "test_metrics.csv"
            )
            metrics = read_metrics(metrics_path)
            elapsed = format_seconds(time.perf_counter() - dataset_start)

            row = {"数据集编号": dataset_name, "总运行耗时": elapsed}
            for output_name, metric_key in METRIC_MAP.items():
                row[output_name] = metrics.get(metric_key, "")
            summary_rows.append(row)
            write_summary(output_csv, summary_rows)

            print_progress(position, total, dataset_name, f"done, elapsed={elapsed}")
    finally:
        restore_config(config_path, original_config)

    print(f"All done. Total elapsed: {format_seconds(time.perf_counter() - all_start)}", flush=True)
    print(f"Saved summary: {output_csv.resolve()}", flush=True)


def main(project: str | None = None) -> None:
    parser = build_parser(default_project=project)
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
