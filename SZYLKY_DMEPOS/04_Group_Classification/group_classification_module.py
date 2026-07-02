from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from hyperparameters import GROUP, RANDOM_SEED, THRESHOLD_GRID
from logging_utils import log_progress, print_table
from screening_evaluation import evaluate_screening, screening_metrics_at_rates, write_screening_evaluation

from advanced_group_classification import build_classifier_pipeline, classifier_name, merge_graphsage_features


os.environ.setdefault("LOKY_MAX_CPU_COUNT", GROUP["loky_max_cpu_count"])

MODEL_FILENAME = "provider_group_classifier.joblib"
DEFAULT_SEEDS = GROUP["ensemble_seeds"]
OOF_FOLDS = GROUP["oof_folds"]
ADAPTIVE_RATE_MULTIPLIER = GROUP["adaptive_rate_multiplier"]
NEGATIVE_SAMPLE_RATIO = GROUP["negative_sample_ratio"]
BALANCED_FOUR_METRIC = GROUP["balanced_metric_name"]
BALANCED_FOUR_COMPONENTS = GROUP["balanced_metric_components"]
SUPPORTED_THRESHOLD_METRICS = {
    "f1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "gmean",
    BALANCED_FOUR_METRIC,
}


def _is_leaky_target_encoded_feature(column: str) -> bool:
    return "train_risk_rate" in column or "_risk_" in column or column.endswith("_risk_mean") or column.endswith("_risk_max")


def _is_unsupported_categorical_feature(column: str) -> bool:
    return column.endswith("_mode")


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False).fillna(0)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _to_model_matrix(x: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        numeric = x.apply(pd.to_numeric, errors="coerce")
        return numeric.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _merge_temporal_provider_features(feats: pd.DataFrame, temporal_root: Path, split: str) -> pd.DataFrame:
    feature_files = [
        temporal_root / split / "provider_temporal_embedding.csv",
        temporal_root / split / "provider_future_risk.csv",
    ]
    merged = feats.copy()
    if "Provider" in merged.columns:
        merged["Provider"] = merged["Provider"].astype(str)
    for path in feature_files:
        extra = _load_csv(path)
        if extra.empty or "Provider" not in extra.columns:
            continue
        extra["Provider"] = extra["Provider"].astype(str)
        extra = extra.drop(columns=["y_true", "ProviderLabel", "PotentialFraud"], errors="ignore")
        value_cols = [c for c in extra.columns if c != "Provider"]
        if not value_cols:
            continue
        prefix = path.stem.replace("provider_", "").replace("_features", "")
        rename_map = {
            c: c if c.startswith(("temporal_", "future_")) else f"{prefix}_{c}"
            for c in value_cols
        }
        merged = merged.merge(extra.rename(columns=rename_map), on="Provider", how="left")
    return merge_graphsage_features(merged.fillna(0), temporal_root, split).fillna(0)


def _select_feature_frame(
    split_dir: Path,
    *,
    temporal_root: str | Path | None = None,
    include_labels: bool = True,
    **_: object,
) -> pd.DataFrame:
    static_df = _load_csv(split_dir / "provider_static_features.csv")
    if static_df.empty:
        return pd.DataFrame()
    static_df["Provider"] = static_df["Provider"].astype(str)
    if "ProviderLabel" in static_df.columns:
        label_df = static_df[["Provider", "ProviderLabel"]].copy()
        feats = static_df.drop(columns=["ProviderLabel"], errors="ignore")
    else:
        feats = static_df
        label_df = _load_csv(split_dir / "provider_labels.csv") if include_labels else pd.DataFrame()
        if not label_df.empty and "Provider" in label_df.columns:
            label_df["Provider"] = label_df["Provider"].astype(str)
    if include_labels and not label_df.empty and "ProviderLabel" not in feats.columns:
        feats = feats.merge(label_df, on="Provider", how="left")
    temporal_root = Path(temporal_root) if temporal_root is not None else split_dir.parent.parent / "03_Temporal_Modeling"
    feats = _merge_temporal_provider_features(feats, temporal_root, split_dir.name)
    return feats.fillna(0)


def _normalize_label_series(label_series: pd.Series) -> pd.Series:
    mapped = pd.Series(label_series).map({"Yes": 1, "No": 0, "Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0})
    if mapped.notna().any():
        return pd.to_numeric(mapped.fillna(label_series), errors="coerce")
    return pd.to_numeric(label_series, errors="coerce")


def _split_xy(df: pd.DataFrame):
    label_col = None
    for candidate in ("ProviderLabel", "PotentialFraud"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Training/test data is missing ProviderLabel or PotentialFraud.")
    y = _normalize_label_series(df[label_col]).fillna(0).astype(int)
    x = df.drop(columns=[label_col], errors="ignore").copy()
    return x, y


def _build_model(random_state: int, y: pd.Series | None = None) -> Pipeline:
    return build_classifier_pipeline(
        random_state,
        y,
        GROUP["hist_gbdt_params"],
    )


def _train_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    progress_label: str = "训练分类 ensemble",
    **_: object,
) -> list[Pipeline]:
    models = []
    seed_values = tuple(seeds)
    for idx, seed in enumerate(seed_values, start=1):
        train_x = x
        train_y = y
        pos_idx = np.flatnonzero(y.to_numpy() == 1)
        neg_idx = np.flatnonzero(y.to_numpy() == 0)
        if len(pos_idx) and len(neg_idx) > len(pos_idx) * NEGATIVE_SAMPLE_RATIO:
            rng = np.random.default_rng(seed)
            sampled_neg = rng.choice(neg_idx, size=len(pos_idx) * NEGATIVE_SAMPLE_RATIO, replace=False)
            sample_idx = np.r_[pos_idx, sampled_neg]
            train_x = x.iloc[sample_idx]
            train_y = y.iloc[sample_idx]
        model = _build_model(seed, train_y)
        model.fit(_to_model_matrix(train_x), train_y)
        models.append(model)
        log_progress(progress_label, idx, len(seed_values), extra=f"seed={seed}")
    return models


def _predict_proba(models: list[Pipeline], x: pd.DataFrame) -> np.ndarray:
    matrix = _to_model_matrix(x)
    return np.mean(np.vstack([model.predict_proba(matrix)[:, 1] for model in models]), axis=0)


def _adaptive_threshold(y_prob: np.ndarray, positive_rate: float, fallback: float) -> float:
    if len(y_prob) == 0 or positive_rate <= 0:
        return fallback
    target_rate = min(0.5, max(1.0 / len(y_prob), positive_rate * ADAPTIVE_RATE_MULTIPLIER))
    quantile_threshold = float(np.quantile(y_prob, max(0.0, 1.0 - target_rate)))
    return min(float(fallback), quantile_threshold)


def _choose_threshold_with_validation(
    x: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    threshold_metric: str,
    oof_folds: int = OOF_FOLDS,
    **_: object,
) -> tuple[float, dict]:
    if y.nunique() < 2 or y.value_counts().min() < 2:
        models = _train_ensemble(x[feature_cols], y, progress_label="阈值验证 train_fallback ensemble")
        y_prob = _predict_proba(models, x[feature_cols])
        threshold, metrics = _best_threshold(y, y_prob, threshold_metric=threshold_metric)
        metrics["threshold_source"] = "train_fallback"
        return threshold, metrics

    n_splits = min(oof_folds, int(y.value_counts().min()))
    oof_prob = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(x[feature_cols], y), start=1):
        fold_models = _train_ensemble(
            x.iloc[train_idx][feature_cols],
            y.iloc[train_idx],
            progress_label=f"OOF 阈值验证 fold {fold_idx}/{n_splits}",
        )
        oof_prob[val_idx] = _predict_proba(fold_models, x.iloc[val_idx][feature_cols])
        log_progress("OOF 阈值验证", fold_idx, n_splits, extra=f"fold={fold_idx}/{n_splits}")
    threshold, metrics = _best_threshold(y, oof_prob, threshold_metric=threshold_metric)
    metrics["threshold_source"] = "oof_validation"
    metrics["validation_size"] = int(len(y))
    metrics["oof_folds"] = int(n_splits)
    return threshold, metrics


def _balanced_four_score(metrics: dict) -> tuple[float, float, float]:
    values = np.array([float(metrics.get(metric, np.nan)) for metric in BALANCED_FOUR_COMPONENTS], dtype=float)
    if np.isnan(values).any():
        return float("nan"), float("nan"), float("nan")
    values = np.clip(values, 0.0, None)
    harmonic = float(len(values) / np.sum(1.0 / np.clip(values, 1e-12, None)))
    spread = float(values.max() - values.min())
    return harmonic - GROUP["balanced_spread_penalty"] * spread, harmonic, spread


def _threshold_score(metrics: dict, threshold_metric: str) -> tuple[float, float]:
    if threshold_metric == BALANCED_FOUR_METRIC:
        score, harmonic, spread = _balanced_four_score(metrics)
        metrics["threshold_balance_hmean"] = harmonic
        metrics["threshold_balance_spread"] = spread
        return score, spread
    return float(metrics[threshold_metric]), 0.0


def _best_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold_metric: str = BALANCED_FOUR_METRIC) -> tuple[float, dict]:
    threshold_metric = threshold_metric.lower()
    if threshold_metric not in SUPPORTED_THRESHOLD_METRICS:
        raise ValueError(f"Unsupported threshold_metric={threshold_metric}. Choose one of {sorted(SUPPORTED_THRESHOLD_METRICS)}.")
    best_thr = 0.5
    best_metrics = None
    best_score = -1.0
    best_spread = float("inf")
    for thr in THRESHOLD_GRID:
        pred = (y_prob >= thr).astype(int)
        metrics = _safe_metrics(y_true, pred, y_prob=y_prob)
        score, spread = _threshold_score(metrics, threshold_metric)
        if np.isnan(score):
            continue
        if (
            score > best_score
            or (
                np.isclose(score, best_score)
                and (
                    spread < best_spread
                    or (np.isclose(spread, best_spread) and metrics["f1"] > (best_metrics or {}).get("f1", -1.0))
                )
            )
        ):
            best_score = score
            best_spread = spread
            best_thr = float(thr)
            best_metrics = metrics
    metrics = best_metrics or _safe_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob=y_prob)
    metrics["threshold_metric"] = threshold_metric
    metrics["threshold_score"] = best_score
    return best_thr, metrics


def _safe_metrics(y_true: pd.Series, y_pred: np.ndarray, *, y_prob: np.ndarray | None = None) -> dict:
    yt = pd.Series(y_true).astype(int)
    if len(yt) == 0 or yt.nunique() < 2:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "precision": float("nan"), "f1": float("nan"), "recall": float("nan"), "specificity": float("nan"), "gmean": float("nan"), "pr_auc": float("nan"), "roc_auc": float("nan"), "tn": float("nan"), "fp": float("nan"), "fn": float("nan"), "tp": float("nan")}
    tn, fp, fn, tp = confusion_matrix(yt, y_pred).ravel()
    recall = float(recall_score(yt, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    metrics = {
        "accuracy": float(accuracy_score(yt, y_pred)),
        "precision": float(precision_score(yt, y_pred, zero_division=0)),
        "f1": float(f1_score(yt, y_pred, zero_division=0)),
        "recall": recall,
        "specificity": specificity,
        "gmean": float(np.sqrt(recall * specificity)) if not np.isnan(specificity) else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(yt, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if y_prob is not None and yt.nunique() > 1:
        metrics["pr_auc"] = float(average_precision_score(yt, y_prob))
        metrics["roc_auc"] = float(roc_auc_score(yt, y_prob))
    else:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")
    return metrics


def run_group_classification(
    preprocess_root: str | Path,
    temporal_root: str | Path,
    output_root: str | Path,
    model_output_root: str | Path | None = None,
    *,
    split: str = "train",
    threshold_metric: str = BALANCED_FOUR_METRIC,
    threshold_override: float | None = None,
) -> Dict[str, object]:
    preprocess_root = Path(preprocess_root)
    split_dir = preprocess_root / split
    out_dir = Path(output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_output_root) if model_output_root else out_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / MODEL_FILENAME

    df = _select_feature_frame(split_dir, temporal_root=temporal_root, include_labels=True)
    if df.empty:
        raise ValueError(f"{split} feature frame is empty. Run preprocessing first.")

    x, y = _split_xy(df)
    feature_cols = [
        c
        for c in x.columns
        if c != "Provider"
        and not _is_leaky_target_encoded_feature(c)
        and not _is_unsupported_categorical_feature(c)
    ]
    if threshold_override is not None:
        threshold = float(threshold_override)
        threshold_metrics = {
            "threshold_source": "manual_override",
            "threshold_metric": threshold_metric,
            "threshold_score": float("nan"),
        }
    else:
        threshold, threshold_metrics = _choose_threshold_with_validation(x, y, feature_cols, threshold_metric)

    models = _train_ensemble(x[feature_cols], y, progress_label="最终分类 ensemble")
    y_prob = _predict_proba(models, x[feature_cols])
    pred = (y_prob >= threshold).astype(int)
    metrics = _safe_metrics(y, pred, y_prob=y_prob)
    metrics["best_threshold"] = threshold
    metrics["threshold_source"] = threshold_metrics.get("threshold_source")
    metrics["threshold_metric"] = threshold_metrics.get("threshold_metric")
    metrics["validation_f1"] = threshold_metrics.get("f1")
    metrics["validation_accuracy"] = threshold_metrics.get("accuracy")
    metrics["validation_balanced_accuracy"] = threshold_metrics.get("balanced_accuracy")
    metrics["validation_precision"] = threshold_metrics.get("precision")
    metrics["validation_recall"] = threshold_metrics.get("recall")
    metrics["validation_gmean"] = threshold_metrics.get("gmean")
    metrics["validation_threshold_score"] = threshold_metrics.get("threshold_score")
    metrics["validation_balance_hmean"] = threshold_metrics.get("threshold_balance_hmean")
    metrics["validation_balance_spread"] = threshold_metrics.get("threshold_balance_spread")
    metrics["validation_pr_auc"] = threshold_metrics.get("pr_auc")
    metrics["validation_roc_auc"] = threshold_metrics.get("roc_auc")
    metrics["validation_size"] = threshold_metrics.get("validation_size")
    metrics["oof_folds"] = threshold_metrics.get("oof_folds")
    metrics["prediction_rate"] = float(np.mean(pred))
    metrics["positive_rate"] = float(np.mean(y))
    classifier = classifier_name(models)
    metrics["classifier"] = classifier

    joblib.dump(
        {
            "models": models,
            "threshold": threshold,
            "feature_cols": feature_cols,
            "threshold_metrics": threshold_metrics,
            "positive_rate": float(np.mean(y)),
            "classifier": classifier,
        },
        model_path,
    )

    pd.DataFrame({"Provider": df["Provider"], "y_true": y, "y_prob": y_prob, "y_pred": pred}).to_csv(out_dir / f"{split}_predictions.csv", index=False, float_format="%.6f")
    pd.DataFrame([metrics]).to_csv(out_dir / f"{split}_metrics.csv", index=False, float_format="%.6f")
    print_table(
        f"Group classification training summary {split}",
        [
            {
                "features": len(feature_cols),
                "classifier": classifier,
                "threshold": threshold,
                "threshold_source": metrics["threshold_source"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "gmean": metrics["gmean"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
            }
        ],
    )
    return {"metrics": metrics, "model_path": str(model_path), "predictions": pred, "threshold": threshold, "best_metrics": threshold_metrics}


def predict_groups(
    preprocess_root: str | Path,
    temporal_root: str | Path,
    output_root: str | Path,
    model_output_root: str | Path,
    split: str = "test",
    threshold_override: float | None = None,
) -> Dict[str, object]:
    preprocess_root = Path(preprocess_root)
    split_dir = preprocess_root / split
    model_path = Path(model_output_root) / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    payload = joblib.load(model_path)
    df = _select_feature_frame(split_dir, temporal_root=temporal_root, include_labels=False)
    if df.empty:
        raise ValueError(f"{split} feature frame is empty.")

    out_dir = Path(output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    models = payload["models"]
    saved_threshold = float(payload["threshold"])
    threshold_metrics = payload.get("threshold_metrics") or {}
    positive_rate = float(payload.get("positive_rate", 0.0))
    feature_cols = payload["feature_cols"]
    x = df.drop(columns=["ProviderLabel", "PotentialFraud"], errors="ignore")
    missing_cols = [col for col in feature_cols if col not in x.columns]
    if missing_cols:
        x = pd.concat(
            [x, pd.DataFrame(0, index=x.index, columns=missing_cols)],
            axis=1,
        )
    y_prob = _predict_proba(models, x[feature_cols])
    if threshold_override is not None:
        threshold = float(threshold_override)
    elif threshold_metrics.get("threshold_source") == "manual_override":
        threshold = saved_threshold
    else:
        threshold = _adaptive_threshold(y_prob, positive_rate, saved_threshold)
    pred = (y_prob >= threshold).astype(int)
    out = pd.DataFrame({"Provider": df["Provider"], "y_prob": y_prob, "y_pred": pred})
    out.to_csv(out_dir / f"{split}_predictions.csv", index=False, float_format="%.6f")

    label_df = _load_csv(split_dir / "provider_labels.csv")
    y_true = None
    if not label_df.empty and {"Provider", "ProviderLabel"}.issubset(label_df.columns):
        label_df["Provider"] = label_df["Provider"].astype(str)
        y_true = out[["Provider"]].merge(label_df[["Provider", "ProviderLabel"]], on="Provider", how="left")["ProviderLabel"]
        y_true = _normalize_label_series(y_true).fillna(0).astype(int)
    metrics = _safe_metrics(y_true, pred, y_prob=y_prob) if y_true is not None else {}
    if metrics:
        metrics["threshold"] = threshold
        classifier = classifier_name(models)
        metrics["classifier"] = classifier
        screening_summary, screening_ranked = evaluate_screening(out, label_df)
        metrics.update(screening_metrics_at_rates(screening_summary))
        pd.DataFrame([metrics]).to_csv(out_dir / f"{split}_metrics.csv", index=False, float_format="%.6f")
        screening_paths = write_screening_evaluation(out, label_df, out_dir, split=split)
        print_table(
            f"Group classification inference summary {split}",
            [
                {
                    "features": len(feature_cols),
                    "classifier": classifier,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "recall": metrics["recall"],
                    "gmean": metrics["gmean"],
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                    "precision@1%": metrics.get("precision_at_1pct"),
                    "recall@1%": metrics.get("recall_at_1pct"),
                    "f1@1%": metrics.get("f1_at_1pct"),
                    "precision@5%": metrics.get("precision_at_5pct"),
                    "recall@5%": metrics.get("recall_at_5pct"),
                    "f1@5%": metrics.get("f1_at_5pct"),
                    "lift@5%": metrics.get("lift_at_5pct"),
                }
            ],
        )
    else:
        screening_paths = {}
    return {"predictions": out, "model_path": str(model_path), "threshold": threshold, "metrics": metrics, "screening_paths": screening_paths}
