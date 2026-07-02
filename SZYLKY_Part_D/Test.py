import argparse
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
from hyperparameters import TEMPORAL
from logging_utils import log_line, log_stage, print_summary, set_verbose
from preprocess_medical_insurance import preprocess_medical_insurance_data
from temporal_rnn_module import build_temporal_outputs
from group_classification_module import predict_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical fraud test entrypoint")
    parser.add_argument("--skip-preprocess", action="store_true", help="Reuse current preprocessing outputs when available")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing outputs even when cache is current")
    parser.add_argument("--reuse-temporal", action="store_true", help="Reuse existing test temporal outputs")
    parser.add_argument("--refresh-temporal", action="store_true", help="Force rebuilding temporal test outputs even when an optimized mainline run is installed")
    parser.add_argument("--threshold-override", type=float, default=None, help="Temporarily override the saved classification threshold")
    parser.add_argument("--run-id", default=None, help="Training run id under 05_Outputs. Defaults to latest_run.txt.")
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
    run_id = args.run_id
    if run_id is None:
        latest_run_path = get_latest_run_path(BASE_DIR)
        if not latest_run_path.exists():
            raise FileNotFoundError(f"Latest run file not found: {latest_run_path}. Run Train.py first or pass --run-id.")
        run_id = latest_run_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError("Empty run id. Run Train.py first or pass --run-id.")

    dataset_root = BASE_DIR / config.dataset_dir
    run_root = get_run_root(BASE_DIR, run_id)
    preprocess_root = run_root / config.preprocess.output_subdir
    temporal_root = run_root / config.temporal.output_subdir
    group_root = run_root / config.group.output_subdir
    model_root = run_root / config.model_output.output_subdir / config.model_output.model_subdir
    optimized_manifest = run_root / "optimized_mainline_manifest.json"
    effective_reuse_temporal = args.reuse_temporal or (optimized_manifest.exists() and not args.refresh_temporal)

    log_line("Test task started")
    log_line(f"Run id: {run_id}")
    log_line(f"Unified output directory: {run_root}")
    log_line(
        "Config: "
        f"skip_preprocess={args.skip_preprocess}, "
        f"force_preprocess={args.force_preprocess}, "
        f"reuse_temporal={effective_reuse_temporal}, "
        f"refresh_temporal={args.refresh_temporal}, "
        f"threshold_override={args.threshold_override}"
    )

    if args.skip_preprocess or (not args.force_preprocess and _preprocess_outputs_current(dataset_root, preprocess_root)):
        log_line("Skipping preprocessing; cached outputs are current", tag="SKIP")
    else:
        with log_stage("Preprocessing Train/Test data"):
            preprocess_medical_insurance_data(dataset_root=dataset_root, output_root=preprocess_root)

    if effective_reuse_temporal:
        reason = "optimized mainline manifest found" if optimized_manifest.exists() and not args.reuse_temporal else "requested by --reuse-temporal"
        log_line(f"Skipping temporal inference; reusing existing outputs ({reason})", tag="SKIP")
    else:
        with log_stage("Temporal inference test"):
            build_temporal_outputs(
                data_root=preprocess_root,
                split="test",
                output_root=temporal_root,
                model_output_root=model_root,
                load_from_train_model=True,
                write_dense_graph=args.write_dense_graph,
                relation_top_k=TEMPORAL["relation_top_k"],
                relation_min_sim=TEMPORAL["relation_min_sim"],
                business_weight_mode=TEMPORAL["business_weight_mode"],
                relation_max_degree=TEMPORAL["relation_max_degree"],
                relation_weight_transform=TEMPORAL["relation_weight_transform"],
            )

    with log_stage("Group classification inference test"):
        result = predict_groups(
            preprocess_root=preprocess_root,
            temporal_root=temporal_root,
            output_root=group_root,
            model_output_root=model_root,
            split="test",
            threshold_override=args.threshold_override,
        )

    metrics = result.get("metrics") or {}
    if metrics:
        print_summary(
            "Test result",
            {
                "Accuracy": metrics.get("accuracy"),
                "Precision": metrics.get("precision"),
                "F1": metrics.get("f1"),
                "Recall": metrics.get("recall"),
                "G-mean": metrics.get("gmean"),
                "PR-AUC": metrics.get("pr_auc"),
                "ROC-AUC": metrics.get("roc_auc"),
                "Precision@1%": metrics.get("precision_at_1pct"),
                "Recall@1%": metrics.get("recall_at_1pct"),
                "F1@1%": metrics.get("f1_at_1pct"),
                "Precision@5%": metrics.get("precision_at_5pct"),
                "Recall@5%": metrics.get("recall_at_5pct"),
                "F1@5%": metrics.get("f1_at_5pct"),
                "Lift@5%": metrics.get("lift_at_5pct"),
                "Threshold": result.get("threshold"),
                "Classifier": metrics.get("classifier"),
            },
        )
        print_summary(
            "Confusion matrix",
            {
                "TN": metrics.get("tn"),
                "FP": metrics.get("fp"),
                "FN": metrics.get("fn"),
                "TP": metrics.get("tp"),
            },
        )
    log_line(f"Model path: {result['model_path']}", tag="DONE")
    log_line(f"Prediction directory: {group_root / 'test'}", tag="DONE")
    log_line("Test task finished", tag="DONE")


if __name__ == "__main__":
    main()
