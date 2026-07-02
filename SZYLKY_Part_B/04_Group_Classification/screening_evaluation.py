from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hyperparameters import SCREENING

DEFAULT_REVIEW_RATES = SCREENING["review_rates"]
DEFAULT_TOP_KS = SCREENING["top_ks"]


def _normalize_label_series(label_series: pd.Series) -> pd.Series:
    mapped = pd.Series(label_series).map({"Yes": 1, "No": 0, "Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0})
    if mapped.notna().any():
        return pd.to_numeric(mapped.fillna(label_series), errors="coerce")
    return pd.to_numeric(label_series, errors="coerce")


def _top_k_metrics(sorted_df: pd.DataFrame, k: int, total_positives: int, total_rows: int, policy_name: str) -> dict:
    k = int(max(1, min(k, total_rows)))
    reviewed = sorted_df.head(k)
    tp = int(reviewed["ProviderLabel"].sum())
    fp = int(k - tp)
    fn = int(total_positives - tp)
    tn = int(total_rows - k - fn)
    precision = tp / max(1, k)
    recall = tp / max(1, total_positives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    base_rate = total_positives / max(1, total_rows)
    lift = precision / max(1e-12, base_rate)
    threshold = float(reviewed["y_prob"].min()) if len(reviewed) else np.nan
    return {
        "policy": policy_name,
        "review_count": k,
        "review_rate": k / max(1, total_rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "lift": lift,
        "captured_positive_count": tp,
        "total_positive_count": total_positives,
    }


def evaluate_screening(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    review_rates: Iterable[float] = DEFAULT_REVIEW_RATES,
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty or labels.empty:
        return pd.DataFrame(), pd.DataFrame()
    pred = predictions.copy()
    lab = labels.copy()
    pred["Provider"] = pred["Provider"].astype(str)
    lab["Provider"] = lab["Provider"].astype(str)
    lab["ProviderLabel"] = _normalize_label_series(lab["ProviderLabel"]).fillna(0).astype(int)
    df = pred.merge(lab[["Provider", "ProviderLabel"]], on="Provider", how="inner")
    if df.empty or "y_prob" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df.sort_values("y_prob", ascending=False).reset_index(drop=True)
    total_rows = len(df)
    total_positives = int(df["ProviderLabel"].sum())
    rows = []
    for rate in review_rates:
        rows.append(_top_k_metrics(df, int(np.ceil(total_rows * float(rate))), total_positives, total_rows, f"top_{float(rate):.1%}"))
    for k in top_ks:
        if int(k) <= total_rows:
            rows.append(_top_k_metrics(df, int(k), total_positives, total_rows, f"top_{int(k)}"))

    ranked = df.copy()
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked["cumulative_tp"] = ranked["ProviderLabel"].cumsum()
    ranked["cumulative_review_rate"] = ranked["rank"] / total_rows
    ranked["cumulative_recall"] = ranked["cumulative_tp"] / max(1, total_positives)
    ranked["cumulative_precision"] = ranked["cumulative_tp"] / ranked["rank"]
    return pd.DataFrame(rows), ranked


def write_screening_evaluation(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    out_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, ranked = evaluate_screening(predictions, labels)
    paths = {
        "summary": str(out_dir / f"{split}_screening_summary.csv"),
        "ranked": str(out_dir / f"{split}_screening_ranked_providers.csv"),
    }
    if not summary.empty:
        summary.to_csv(paths["summary"], index=False, float_format="%.6f")
    if not ranked.empty:
        ranked.to_csv(paths["ranked"], index=False, float_format="%.6f")
    return paths


def screening_metrics_at_rates(summary: pd.DataFrame) -> dict[str, float]:
    if summary.empty:
        return {}
    rows = {}
    for _, row in summary.iterrows():
        policy = str(row.get("policy", ""))
        if policy == "top_1.0%":
            rows["tp_at_1pct"] = float(row["tp"])
            rows["precision_at_1pct"] = float(row["precision_at_k"])
            rows["recall_at_1pct"] = float(row["recall_at_k"])
            rows["f1_at_1pct"] = float(row["f1_at_k"])
        elif policy == "top_5.0%":
            rows["tp_at_5pct"] = float(row["tp"])
            rows["precision_at_5pct"] = float(row["precision_at_k"])
            rows["recall_at_5pct"] = float(row["recall_at_k"])
            rows["f1_at_5pct"] = float(row["f1_at_k"])
            rows["lift_at_5pct"] = float(row["lift"])
    return rows
