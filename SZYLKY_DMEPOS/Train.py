import argparse
from datetime import datetime
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
for subdir in [
    "00_Global_Config",
    "02_Data_Preprocessing",
    "03_Temporal_Modeling",
    "04_Group_Classification",
]:
    full = BASE_DIR / subdir
    if str(full) not in sys.path:
        sys.path.insert(0, str(full))

from config import get_default_config, get_latest_run_path, get_run_root
from hyperparameters import GROUP, TEMPORAL
from logging_utils import log_line, log_stage, print_summary, set_verbose
from preprocess_medical_insurance import preprocess_medical_insurance_data
from temporal_rnn_module import build_temporal_outputs
from group_classification_module import SUPPORTED_THRESHOLD_METRICS, run_group_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical fraud training entrypoint")
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse current preprocessing outputs when available")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing outputs even when cache is current")
    parser.add_argument("--reuse-temporal", action="store_true", help="Reuse existing temporal outputs")
    parser.add_argument(
        "--threshold-override",
        type=float,
        default=GROUP["fixed_threshold"],
        help="Final classification threshold. Defaults to the optimized fixed threshold.",
    )
    parser.add_argument("--tune-threshold", action="store_true", help="Run OOF threshold search instead of using the fixed threshold.")
    parser.add_argument(
        "--threshold-metric",
        default=GROUP["balanced_metric_name"],
        choices=sorted(SUPPORTED_THRESHOLD_METRICS),
        help="OOF metric used to choose the classification threshold",
    )
    parser.add_argument("--run-id", default=None, help="Output run id. Defaults to YYYYMMDDHHMM.")
    parser.add_argument("--write-dense-graph", action="store_true", help="Also write the dense provider graph CSV")
    parser.add_argument("--verbose", action="store_true", help="Show detailed tables and progress")
    return parser.parse_args()


def _preprocess_outputs_current(dataset_root: Path, preprocess_root: Path) -> bool:
    required = [
        preprocess_root / split / name
        for split in ("train", "test")
        for name in ("provider_static_features.csv", "provider_temporal_features.csv", "provider_labels.csv")
    ]
    if not all(path.exists() for path in required):
        return False
    source_files = list(dataset_root.rglob("*.csv"))
    if not source_files:
        return False
    newest_source = max(path.stat().st_mtime for path in source_files)
    oldest_output = min(path.stat().st_mtime for path in required)
    return oldest_output >= newest_source


def main() -> None:
    args = parse_args()
    set_verbose(args.verbose)
    config = get_default_config()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d%H%M")
    dataset_root = BASE_DIR / config.dataset_dir
    run_root = get_run_root(BASE_DIR, run_id)
    preprocess_root = run_root / config.preprocess.output_subdir
    temporal_root = run_root / config.temporal.output_subdir
    group_root = run_root / config.group.output_subdir
    model_root = run_root / config.model_output.output_subdir / config.model_output.model_subdir
    run_root.mkdir(parents=True, exist_ok=True)
    latest_run_path = get_latest_run_path(BASE_DIR)
    latest_run_path.parent.mkdir(parents=True, exist_ok=True)
    latest_run_path.write_text(run_id, encoding="utf-8")

    log_line("Training task started")
    log_line(f"Run id: {run_id}")
    log_line(f"Unified output directory: {run_root}")
    effective_threshold = None if args.tune_threshold else args.threshold_override
    log_line(
        "Config: "
        f"skip_preprocess={args.skip_preprocess}, "
        f"force_preprocess={args.force_preprocess}, "
        f"reuse_temporal={args.reuse_temporal}, "
        f"threshold_override={effective_threshold}, "
        f"tune_threshold={args.tune_threshold}, "
        f"threshold_metric={args.threshold_metric}"
    )

    if args.skip_preprocess or (not args.force_preprocess and _preprocess_outputs_current(dataset_root, preprocess_root)):
        log_line("Skipping preprocessing; cached outputs are current", tag="SKIP")
    else:
        with log_stage("Preprocessing Train/Test data"):
            preprocess_medical_insurance_data(dataset_root=dataset_root, output_root=preprocess_root)

    if args.reuse_temporal:
        log_line("Skipping temporal modeling; reusing existing outputs", tag="SKIP")
    else:
        with log_stage("Temporal modeling train"):
            build_temporal_outputs(
                data_root=preprocess_root,
                split="train",
                output_root=temporal_root,
                model_output_root=model_root,
                load_from_train_model=False,
                write_dense_graph=args.write_dense_graph,
                relation_top_k=TEMPORAL["relation_top_k"],
                relation_min_sim=TEMPORAL["relation_min_sim"],
                business_weight_mode=TEMPORAL["business_weight_mode"],
                relation_max_degree=TEMPORAL["relation_max_degree"],
                relation_weight_transform=TEMPORAL["relation_weight_transform"],
            )

    with log_stage("Group classification train"):
        result = run_group_classification(
            preprocess_root=preprocess_root,
            temporal_root=temporal_root,
            output_root=group_root,
            model_output_root=model_root,
            split="train",
            threshold_metric=args.threshold_metric,
            threshold_override=effective_threshold,
        )

    metrics = result["metrics"]
    print_summary(
        "Training result",
        {
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "F1": metrics["f1"],
            "Recall": metrics["recall"],
            "G-mean": metrics["gmean"],
            "PR-AUC": metrics["pr_auc"],
            "ROC-AUC": metrics["roc_auc"],
            "Threshold": metrics["best_threshold"],
            "Threshold metric": metrics["threshold_metric"],
            "Classifier": metrics["classifier"],
            "Threshold source": metrics["threshold_source"],
        },
    )
    log_line(f"Model saved: {result['model_path']}", tag="DONE")
    log_line(f"Run output saved: {run_root}", tag="DONE")
    log_line("Training task finished", tag="DONE")


if __name__ == "__main__":
    main()
