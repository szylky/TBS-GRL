from __future__ import annotations

"""Provider-centric preprocessing for the medical fraud dataset.

This pipeline builds three kinds of provider-level artifacts:
1) static features aggregated from claims and beneficiary tables
2) temporal monthly features aggregated by Provider + TimeWindow
3) optional graph features derived from shared-beneficiary collaboration edges

It is designed to be resilient: if some source files are missing, it will still
produce the artifacts that can be built from the available inputs.
"""

from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parents[1]
for subdir in [BASE_DIR, BASE_DIR / "00_Global_Config"]:
    if str(subdir) not in sys.path:
        sys.path.insert(0, str(subdir))

from hyperparameters import PREPROCESSING, RANDOM_SEED
from logging_utils import log_line, log_progress, log_stage, print_table


def _safe_read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    path = Path(path) if path is not None else None
    if path is None or path.name.startswith("._"):
        return pd.DataFrame()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs).fillna("")
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="ignore", low_memory=False, **kwargs).fillna("")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _find_first(files: Iterable[Path], keywords: list[str]) -> Path | None:
    for f in files:
        name = f.name.lower()
        if all(k.lower() in name for k in keywords):
            return f
    return None


def _find_provider_file(files: Iterable[Path], split: str) -> Path | None:
    split_lower = split.lower()
    priority_keywords = [
        [split_lower, "list"],
        [split_lower, "provider"],
        [split_lower, "potentialfraud"],
        [split_lower, "train"],
        [split_lower, "test"],
    ]
    for keywords in priority_keywords:
        match = _find_first(files, keywords)
        if match is not None:
            return match
    for f in files:
        name = f.name.lower()
        if split_lower in name and all(x not in name for x in ["beneficiary", "inpatient", "outpatient"]):
            return f
    return None


def _list_csvs(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted([p for p in split_dir.rglob("*.csv") if p.is_file() and not p.name.startswith("._")])


def _split_source_csvs(dataset_root: Path, split: str) -> list[Path]:
    split_dir = dataset_root / split
    csvs = _list_csvs(split_dir)
    if csvs:
        return csvs
    for stem in (f"Part_D_{split}", f"DMEPOS_{split}", split):
        direct = dataset_root / f"{stem}.csv"
        if direct.exists():
            return [direct]
    split_lower = split.lower()
    matches = [
        p
        for p in dataset_root.glob("*.csv")
        if split_lower in p.stem.lower() and not p.name.startswith("._")
    ]
    return sorted(matches)


def _normalize_label_series(label_series: pd.Series) -> pd.Series:
    normalized = pd.Series(label_series).astype(str).str.strip().str.lower()
    mapped = normalized.map(
        {
            "yes": 1,
            "y": 1,
            "true": 1,
            "1": 1,
            "是": 1,
            "诈骗": 1,
            "欺诈": 1,
            "否": 0,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
            "非诈骗": 0,
            "非欺诈": 0,
        }
    )
    return pd.to_numeric(mapped.where(mapped.notna(), label_series), errors="coerce")


def _is_dmepos_flat(df: pd.DataFrame) -> bool:
    return {"Rfrg_NPI", "Year"}.issubset(df.columns) or {"Prscrbr_NPI", "Year"}.issubset(df.columns)


def _is_dmepos_flat_file(path: Path) -> bool:
    return _is_dmepos_flat(_safe_read_csv(path, nrows=0))


def _prepare_dmepos_flat(df: pd.DataFrame, *, keep_label_in_claims: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["Provider", "ProviderLabel"])

    out = df.copy()
    provider_col = "Rfrg_NPI" if "Rfrg_NPI" in out.columns else "Prscrbr_NPI"
    out["Provider"] = out[provider_col].astype(str)
    out["ClaimID"] = np.arange(len(out)).astype(str)
    out["BeneID"] = out["Provider"] + "_" + out["ClaimID"]
    out["ClaimType"] = "part_d" if provider_col == "Prscrbr_NPI" else "dmepos"
    out["TimeWindow"] = "TW_" + pd.to_numeric(out["Year"], errors="coerce").fillna(0).astype(int).astype(str)

    if "Tot_Drug_Cst" in out.columns:
        amount_col = "Tot_Drug_Cst"
    else:
        amount_col = "Avg_Suplr_Mdcr_Pymt_Amt" if "Avg_Suplr_Mdcr_Pymt_Amt" in out.columns else "Avg_Suplr_Mdcr_Alowd_Amt"
    if "Tot_30day_Fills" in out.columns:
        service_col = "Tot_30day_Fills"
    elif "Tot_Suplr_Srvcs" in out.columns:
        service_col = "Tot_Suplr_Srvcs"
    else:
        service_col = "Tot_Suplr_Clms"
    out["_service_count"] = pd.to_numeric(out.get(service_col, 1), errors="coerce").fillna(1).clip(lower=1)
    amount = pd.to_numeric(out.get(amount_col, 0), errors="coerce").fillna(0)
    out["InscClaimAmtReimbursed"] = amount if amount_col == "Tot_Drug_Cst" else amount * out["_service_count"]
    out["DeductibleAmtPaid"] = 0.0
    out["ClaimDurationDays"] = (
        _safe_divide(out["Tot_Day_Suply"], out["Tot_Clms"]).clip(lower=1.0)
        if {"Tot_Day_Suply", "Tot_Clms"}.issubset(out.columns)
        else 1.0
    )
    out["DailyReimbursedAmt"] = _safe_divide(out["InscClaimAmtReimbursed"], out["ClaimDurationDays"])
    out["DiagnosisCodeCount"] = 0.0
    out["ProcedureCodeCount"] = 0.0
    out["Physician_count"] = 0
    out["AdmissionDurationDays"] = 0
    out["HasOperatingPhysician"] = 0
    out["HasOtherPhysician"] = 0
    out["HasAdmitDiagnosis"] = 0
    out["Age"] = 0.0
    out["IsDeceased"] = 0.0
    out["RenalDiseaseIndicator"] = 0.0

    label_col = next(
        (
            col
            for col in ["是否诈骗", "是否欺诈", "ProviderLabel", "PotentialFraud"]
            if col in out.columns
        ),
        None,
    )
    if label_col:
        label_df = (
            out[["Provider", label_col]]
            .assign(ProviderLabel=lambda x: _normalize_label_series(x[label_col]))
            .groupby("Provider", as_index=False)["ProviderLabel"]
            .max()
        )
        if keep_label_in_claims:
            out["ProviderLabel"] = _normalize_label_series(out[label_col]).fillna(0).astype(int)
        if not keep_label_in_claims:
            out = out.drop(columns=[label_col], errors="ignore")
    else:
        label_df = pd.DataFrame(columns=["Provider", "ProviderLabel"])

    return out, label_df.drop_duplicates()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _weighted_entropy(values: pd.Series) -> float:
    total = float(pd.to_numeric(values, errors="coerce").fillna(0.0).sum())
    if total <= 0:
        return 0.0
    p = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _add_binary_presence_features(frame: pd.DataFrame, columns: list[str], prefix: str) -> pd.DataFrame:
    out = frame.copy()
    present_cols = [c for c in columns if c in out.columns]
    if present_cols:
        out[f"{prefix}_count"] = out[present_cols].replace("", np.nan).notna().sum(axis=1)
    else:
        out[f"{prefix}_count"] = 0
    return out


def _add_dmepos_ratio_columns(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return claims
    out = claims.copy()
    if {"Avg_Suplr_Sbmtd_Chrg", "Avg_Suplr_Mdcr_Alowd_Amt"}.issubset(out.columns):
        out["ratio_submitted_to_allowed"] = _safe_divide(out["Avg_Suplr_Sbmtd_Chrg"], out["Avg_Suplr_Mdcr_Alowd_Amt"])
    if {"Avg_Suplr_Mdcr_Pymt_Amt", "Avg_Suplr_Mdcr_Alowd_Amt"}.issubset(out.columns):
        out["ratio_payment_to_allowed"] = _safe_divide(out["Avg_Suplr_Mdcr_Pymt_Amt"], out["Avg_Suplr_Mdcr_Alowd_Amt"])
    if {"Avg_Suplr_Mdcr_Stdzd_Amt", "Avg_Suplr_Mdcr_Pymt_Amt"}.issubset(out.columns):
        out["ratio_std_to_payment"] = _safe_divide(out["Avg_Suplr_Mdcr_Stdzd_Amt"], out["Avg_Suplr_Mdcr_Pymt_Amt"])
    if {"Tot_Suplr_Srvcs", "Tot_Suplr_Clms"}.issubset(out.columns):
        out["ratio_services_to_claims"] = _safe_divide(out["Tot_Suplr_Srvcs"], out["Tot_Suplr_Clms"])
    if {"Tot_Suplr_Benes", "Tot_Suplr_Clms"}.issubset(out.columns):
        out["ratio_benes_to_claims"] = _safe_divide(out["Tot_Suplr_Benes"], out["Tot_Suplr_Clms"])
    if {"Tot_Drug_Cst", "Tot_Clms"}.issubset(out.columns):
        out["ratio_drug_cost_per_claim"] = _safe_divide(out["Tot_Drug_Cst"], out["Tot_Clms"])
    if {"Tot_Drug_Cst", "Tot_30day_Fills"}.issubset(out.columns):
        out["ratio_drug_cost_per_30day_fill"] = _safe_divide(out["Tot_Drug_Cst"], out["Tot_30day_Fills"])
    if {"Tot_Day_Suply", "Tot_Clms"}.issubset(out.columns):
        out["ratio_day_supply_per_claim"] = _safe_divide(out["Tot_Day_Suply"], out["Tot_Clms"])
    if {"GE65_Tot_Clms", "Tot_Clms"}.issubset(out.columns):
        out["ratio_ge65_claim_share"] = _safe_divide(out["GE65_Tot_Clms"], out["Tot_Clms"])
    if {"GE65_Tot_Drug_Cst", "Tot_Drug_Cst"}.issubset(out.columns):
        out["ratio_ge65_drug_cost_share"] = _safe_divide(out["GE65_Tot_Drug_Cst"], out["Tot_Drug_Cst"])
    if {"GE65_Tot_Day_Suply", "Tot_Day_Suply"}.issubset(out.columns):
        out["ratio_ge65_day_supply_share"] = _safe_divide(out["GE65_Tot_Day_Suply"], out["Tot_Day_Suply"])
    amount_cols = [
        c
        for c in [
            "Avg_Suplr_Sbmtd_Chrg",
            "Avg_Suplr_Mdcr_Alowd_Amt",
            "Avg_Suplr_Mdcr_Pymt_Amt",
            "Avg_Suplr_Mdcr_Stdzd_Amt",
        ]
        if c in out.columns
    ]
    count_cols = [c for c in ["Tot_Suplrs", "Tot_Suplr_Clms", "Tot_Suplr_Srvcs"] if c in out.columns]
    for amount_col in amount_cols:
        amount = pd.to_numeric(out[amount_col], errors="coerce").fillna(0.0)
        for count_col in count_cols:
            count = pd.to_numeric(out[count_col], errors="coerce").replace(0, np.nan)
            out[f"ratio_{amount_col}_per_{count_col}"] = (amount / count).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if {"Avg_Suplr_Mdcr_Alowd_Amt", "Avg_Suplr_Sbmtd_Chrg"}.issubset(out.columns):
        out["ratio_Avg_Suplr_Mdcr_Alowd_Amt_per_Avg_Suplr_Sbmtd_Chrg"] = _safe_divide(out["Avg_Suplr_Mdcr_Alowd_Amt"], out["Avg_Suplr_Sbmtd_Chrg"])
    if {"Avg_Suplr_Mdcr_Pymt_Amt", "Avg_Suplr_Sbmtd_Chrg"}.issubset(out.columns):
        out["ratio_Avg_Suplr_Mdcr_Pymt_Amt_per_Avg_Suplr_Sbmtd_Chrg"] = _safe_divide(out["Avg_Suplr_Mdcr_Pymt_Amt"], out["Avg_Suplr_Sbmtd_Chrg"])
    if {"Avg_Suplr_Mdcr_Stdzd_Amt", "Avg_Suplr_Sbmtd_Chrg"}.issubset(out.columns):
        out["ratio_Avg_Suplr_Mdcr_Stdzd_Amt_per_Avg_Suplr_Sbmtd_Chrg"] = _safe_divide(out["Avg_Suplr_Mdcr_Stdzd_Amt"], out["Avg_Suplr_Sbmtd_Chrg"])
    return out


def _dmepos_numeric_columns(claims: pd.DataFrame) -> list[str]:
    preferred = [
        "Prscrbr_State_FIPS",
        "Tot_Clms",
        "Tot_30day_Fills",
        "Tot_Day_Suply",
        "Tot_Drug_Cst",
        "Tot_Benes",
        "GE65_Tot_Clms",
        "GE65_Tot_30day_Fills",
        "GE65_Tot_Drug_Cst",
        "GE65_Tot_Day_Suply",
        "GE65_Tot_Benes",
        "Rfrg_Prvdr_RUCA",
        "Tot_Suplrs",
        "Tot_Suplr_Benes",
        "Tot_Suplr_Clms",
        "Tot_Suplr_Srvcs",
        "Avg_Suplr_Sbmtd_Chrg",
        "Avg_Suplr_Mdcr_Alowd_Amt",
        "Avg_Suplr_Mdcr_Pymt_Amt",
        "Avg_Suplr_Mdcr_Stdzd_Amt",
        "Year",
    ]
    return [
        c
        for c in preferred + [c for c in claims.columns if c.startswith("ratio_")]
        if c in claims.columns
    ]


def _dmepos_categorical_columns(claims: pd.DataFrame) -> list[str]:
    preferred = [
        "Prscrbr_City",
        "Prscrbr_State_Abrvtn",
        "Prscrbr_Type",
        "Prscrbr_Type_Src",
        "Brnd_Name",
        "Gnrc_Name",
        "GE65_Sprsn_Flag",
        "GE65_Bene_Sprsn_Flag",
        "来源文件",
        "HCPCS_Desc",
        "HCPCS_CD",
        "Rfrg_Prvdr_Spclty_Cd",
        "Rfrg_Prvdr_Spclty_Desc",
        "Rfrg_Prvdr_State_FIPS",
        "Rfrg_Prvdr_State_Abrvtn",
        "RBCS_Id",
        "RBCS_Desc",
        "RBCS_Lvl",
        "Rfrg_Prvdr_Spclty_Srce",
        "Rfrg_Prvdr_Ent_Cd",
        "来源文件",
        "Rfrg_Prvdr_City",
        "Rfrg_Prvdr_RUCA_Desc",
        "Rfrg_Prvdr_RUCA_Cat",
        "Suplr_Rentl_Ind",
        "Rfrg_Prvdr_Cntry",
        "Year",
    ]
    return [c for c in preferred if c in claims.columns]


def _dmepos_risk_maps(train_claims: pd.DataFrame, categorical_cols: list[str], smoothing: float = 35.0) -> tuple[dict[str, dict[str, float]], float]:
    if train_claims.empty or "ProviderLabel" not in train_claims.columns:
        return {}, 0.0
    labels = pd.to_numeric(train_claims["ProviderLabel"], errors="coerce").fillna(0).astype(int)
    global_rate = float(labels.mean()) if len(labels) else 0.0
    maps: dict[str, dict[str, float]] = {}
    for col in categorical_cols:
        values = train_claims[col].astype(str).str.strip().replace("", "__MISSING__")
        stats = labels.groupby(values).agg(["mean", "count"])
        if stats.empty:
            continue
        smoothed = (stats["mean"] * stats["count"] + global_rate * smoothing) / (stats["count"] + smoothing)
        maps[col] = smoothed.to_dict()
    return maps, global_rate


def _dmepos_frequency_maps(train_claims: pd.DataFrame, categorical_cols: list[str]) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    if train_claims.empty:
        return maps
    for col in categorical_cols:
        if col not in train_claims.columns:
            continue
        values = train_claims[col].astype(str).str.strip().replace("", "__MISSING__")
        freq = values.value_counts(normalize=True, dropna=False)
        maps[col] = freq.to_dict()
    return maps


def _dmepos_oof_risk_values(train_claims: pd.DataFrame, categorical_cols: list[str], global_rate: float, smoothing: float = 35.0) -> dict[str, pd.Series]:
    if train_claims.empty or "ProviderLabel" not in train_claims.columns:
        return {}
    y = pd.to_numeric(train_claims["ProviderLabel"], errors="coerce").fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
        return {}
    n_splits = min(5, int(np.bincount(y).min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    out: dict[str, pd.Series] = {}
    risk_cols = [
        c
        for c in [
            "HCPCS_CD",
            "RBCS_Id",
            "RBCS_Lvl",
            "Brnd_Name",
            "Gnrc_Name",
            "Prscrbr_Type",
            "Prscrbr_State_Abrvtn",
            "Rfrg_Prvdr_Spclty_Cd",
            "Rfrg_Prvdr_Spclty_Desc",
            "Rfrg_Prvdr_State_Abrvtn",
            "Rfrg_Prvdr_State_FIPS",
            "Suplr_Rentl_Ind",
            "Year",
        ]
        if c in categorical_cols
    ]
    for col in risk_cols:
        values = train_claims[col].astype(str).str.strip().replace("", "__MISSING__")
        risk_values = pd.Series(global_rate, index=train_claims.index, dtype=float)
        for train_idx, val_idx in splitter.split(train_claims, y):
            train_values = values.iloc[train_idx]
            train_y = pd.Series(y[train_idx], index=train_values.index)
            stats = train_y.groupby(train_values).agg(["mean", "count"])
            smoothed = (stats["mean"] * stats["count"] + global_rate * smoothing) / (stats["count"] + smoothing)
            risk_values.iloc[val_idx] = values.iloc[val_idx].map(smoothed).fillna(global_rate).astype(float)
        out[col] = risk_values
    return out


def _dmepos_adaptive_provider_features(
    claims: pd.DataFrame,
    label_df: pd.DataFrame,
    categorical_risk_maps: dict[str, dict[str, float]],
    global_risk: float,
    categorical_frequency_maps: dict[str, dict[str, float]] | None = None,
    *,
    split_key: str,
) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame()
    if "Prscrbr_NPI" in claims.columns:
        return _part_d_provider_features(
            claims,
            label_df,
            categorical_risk_maps,
            global_risk,
            categorical_frequency_maps,
        )
    work = claims.copy()
    work["Provider"] = work["Provider"].astype(str)
    numeric_cols = _dmepos_numeric_columns(work)
    categorical_cols = _dmepos_categorical_columns(work)
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    grouped = work.groupby("Provider", sort=True)
    result = pd.DataFrame({"Provider": grouped.size().index.astype(str)})
    result["row_count"] = grouped.size().to_numpy(dtype=float)
    if numeric_cols:
        numeric_agg = grouped[numeric_cols].agg(["sum", "mean", "median", "std", "min", "max"]).fillna(0.0)
        numeric_agg.columns = [f"{col}_{stat}" for col, stat in numeric_agg.columns]
        result = result.merge(numeric_agg.reset_index(), on="Provider", how="left")
        for col in numeric_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            min_col = f"{col}_min"
            max_col = f"{col}_max"
            if mean_col in result.columns and std_col in result.columns:
                result[f"{col}_cv"] = _safe_divide(result[std_col], result[mean_col].abs())
            if min_col in result.columns and max_col in result.columns:
                max_values = pd.to_numeric(result[max_col], errors="coerce").fillna(0.0)
                min_values = pd.to_numeric(result[min_col], errors="coerce").fillna(0.0)
                result[f"{col}_range"] = max_values - min_values

    risk_values_by_col = _dmepos_oof_risk_values(work, categorical_cols, global_risk) if split_key == "train" else {}
    cat_rows = []
    for provider, group in grouped:
        row = {"Provider": str(provider)}
        n = max(1, len(group))
        for col in categorical_cols:
            values = group[col].astype(str).str.strip().replace("", "__MISSING__")
            counts = values.value_counts(dropna=False)
            row[f"{col}_mode"] = "__MISSING__" if counts.empty else str(counts.index[0])
            row[f"{col}_nunique"] = float(values.nunique(dropna=True))
            row[f"{col}_top_share"] = float(counts.iloc[0] / n) if not counts.empty else 0.0
            entropy = _weighted_entropy(counts) if not counts.empty else 0.0
            nunique = max(1.0, float(values.nunique(dropna=True)))
            row[f"{col}_entropy"] = entropy
            row[f"{col}_normalized_entropy"] = float(entropy / np.log(nunique)) if nunique > 1 else 0.0
            row[f"{col}_nunique_per_row"] = float(nunique / n)
            frequency_map = (categorical_frequency_maps or {}).get(col, {})
            if frequency_map:
                freq = values.map(frequency_map).fillna(0.0).astype(float)
                row[f"{col}_freq_mean"] = float(freq.mean()) if len(freq) else 0.0
                row[f"{col}_freq_max"] = float(freq.max()) if len(freq) else 0.0
                row[f"{col}_freq_min"] = float(freq.min()) if len(freq) else 0.0
                row[f"{col}_rare_share"] = float((freq <= 0.001).mean()) if len(freq) else 0.0
            if split_key == "train" and col in risk_values_by_col:
                risk = risk_values_by_col[col].loc[group.index]
            else:
                mapping = categorical_risk_maps.get(col, {})
                risk = values.map(mapping).fillna(global_risk).astype(float)
            if col in risk_values_by_col or categorical_risk_maps.get(col):
                row[f"{col}_risk_mean"] = float(risk.mean()) if len(risk) else global_risk
                row[f"{col}_risk_max"] = float(risk.max()) if len(risk) else global_risk
        cat_rows.append(row)
    if cat_rows:
        result = result.merge(pd.DataFrame(cat_rows), on="Provider", how="left")

    if "Year" in work.columns and numeric_cols:
        trend_cols = [
            c
            for c in [
                "Tot_Suplrs",
                "Tot_Suplr_Benes",
                "Tot_Suplr_Clms",
                "Tot_Suplr_Srvcs",
                "Tot_Clms",
                "Tot_30day_Fills",
                "Tot_Day_Suply",
                "Tot_Drug_Cst",
                "Tot_Benes",
                "GE65_Tot_Clms",
                "GE65_Tot_Drug_Cst",
                "GE65_Tot_Day_Suply",
                "Avg_Suplr_Sbmtd_Chrg",
                "Avg_Suplr_Mdcr_Alowd_Amt",
                "Avg_Suplr_Mdcr_Pymt_Amt",
                "Avg_Suplr_Mdcr_Stdzd_Amt",
            ]
            if c in numeric_cols
        ]
        yearly = work[["Provider", "Year"] + trend_cols].copy()
        yearly["Year"] = pd.to_numeric(yearly["Year"], errors="coerce")
        yearly = yearly.dropna(subset=["Year"]).groupby(["Provider", "Year"], as_index=False)[trend_cols].mean()
        trend_rows = []
        for provider, group in yearly.groupby("Provider", sort=False):
            row = {"Provider": str(provider), "time_span": float(group["Year"].max() - group["Year"].min())}
            recent = group[group["Year"] == group["Year"].max()]
            for col in trend_cols:
                row[f"{col}_slope"] = _slope(group[col], group["Year"])
                row[f"{col}_recent"] = float(recent[col].mean()) if not recent.empty else 0.0
            trend_rows.append(row)
        if trend_rows:
            result = result.merge(pd.DataFrame(trend_rows), on="Provider", how="left")

    if not label_df.empty and {"Provider", "ProviderLabel"}.issubset(label_df.columns):
        labels = label_df[["Provider", "ProviderLabel"]].copy()
        labels["Provider"] = labels["Provider"].astype(str)
        result = result.merge(labels, on="Provider", how="left")
    return result.fillna(0)


def _part_d_provider_features(
    claims: pd.DataFrame,
    label_df: pd.DataFrame,
    categorical_risk_maps: dict[str, dict[str, float]],
    global_risk: float,
    categorical_frequency_maps: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    work = claims.copy()
    work["Provider"] = work["Provider"].astype(str)
    numeric_cols = [c for c in _dmepos_numeric_columns(work) if c != "Year"]
    categorical_cols = [
        c
        for c in _dmepos_categorical_columns(work)
        if c != "Year" and c != "来源文件"
    ]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    grouped = work.groupby("Provider", sort=True)
    result = pd.DataFrame({"Provider": grouped.size().index.astype(str)})
    result["row_count"] = grouped.size().to_numpy(dtype=float)

    if numeric_cols:
        numeric_agg = grouped[numeric_cols].agg(["sum", "mean", "std", "min", "max"]).fillna(0.0)
        numeric_agg.columns = [f"{col}_{stat}" for col, stat in numeric_agg.columns]
        result = result.merge(numeric_agg.reset_index(), on="Provider", how="left")
        for col in numeric_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            min_col = f"{col}_min"
            max_col = f"{col}_max"
            if mean_col in result.columns and std_col in result.columns:
                result[f"{col}_cv"] = _safe_divide(result[std_col], result[mean_col].abs())
            if min_col in result.columns and max_col in result.columns:
                result[f"{col}_range"] = pd.to_numeric(result[max_col], errors="coerce").fillna(0.0) - pd.to_numeric(result[min_col], errors="coerce").fillna(0.0)

    provider_index = result["Provider"]
    for col in categorical_cols:
        values = work[["Provider", col]].copy()
        values[col] = values[col].astype(str).str.strip().replace("", "__MISSING__")
        cat = pd.DataFrame({"Provider": provider_index})
        cat[f"{col}_nunique"] = values.groupby("Provider")[col].nunique().reindex(provider_index).fillna(0).to_numpy(dtype=float)

        risk_map = categorical_risk_maps.get(col, {})
        if risk_map:
            risk = values[col].map(risk_map).fillna(global_risk).astype(float)
            risk_stats = risk.groupby(values["Provider"]).agg(["mean", "max"]).reindex(provider_index).fillna(global_risk)
            cat[f"{col}_risk_mean"] = risk_stats["mean"].to_numpy(dtype=float)
            cat[f"{col}_risk_max"] = risk_stats["max"].to_numpy(dtype=float)
        result = result.merge(cat, on="Provider", how="left")

    part_d_ratio_pairs = [
        ("partd_claims_per_bene", "Tot_Clms_sum", "Tot_Benes_sum"),
        ("partd_cost_per_bene", "Tot_Drug_Cst_sum", "Tot_Benes_sum"),
        ("partd_days_per_bene", "Tot_Day_Suply_sum", "Tot_Benes_sum"),
        ("partd_fills_per_bene", "Tot_30day_Fills_sum", "Tot_Benes_sum"),
        ("partd_cost_per_claim", "Tot_Drug_Cst_sum", "Tot_Clms_sum"),
        ("partd_days_per_claim", "Tot_Day_Suply_sum", "Tot_Clms_sum"),
        ("partd_fills_per_claim", "Tot_30day_Fills_sum", "Tot_Clms_sum"),
        ("partd_claims_per_generic", "Tot_Clms_sum", "Gnrc_Name_nunique"),
        ("partd_cost_per_generic", "Tot_Drug_Cst_sum", "Gnrc_Name_nunique"),
        ("partd_claims_per_brand", "Tot_Clms_sum", "Brnd_Name_nunique"),
        ("partd_cost_per_brand", "Tot_Drug_Cst_sum", "Brnd_Name_nunique"),
        ("partd_rows_per_generic", "row_count", "Gnrc_Name_nunique"),
        ("partd_rows_per_brand", "row_count", "Brnd_Name_nunique"),
        ("partd_brands_per_generic", "Brnd_Name_nunique", "Gnrc_Name_nunique"),
        ("partd_generics_per_bene", "Gnrc_Name_nunique", "Tot_Benes_sum"),
        ("partd_brands_per_bene", "Brnd_Name_nunique", "Tot_Benes_sum"),
        ("partd_ge65_claims_per_total", "GE65_Tot_Clms_sum", "Tot_Clms_sum"),
        ("partd_ge65_cost_per_total", "GE65_Tot_Drug_Cst_sum", "Tot_Drug_Cst_sum"),
        ("partd_ge65_days_per_total", "GE65_Tot_Day_Suply_sum", "Tot_Day_Suply_sum"),
        ("partd_ge65_benes_per_total", "GE65_Tot_Benes_sum", "Tot_Benes_sum"),
    ]
    for new_col, numerator, denominator in part_d_ratio_pairs:
        if numerator in result.columns and denominator in result.columns:
            result[new_col] = _safe_divide(result[numerator], result[denominator])

    for col in [
        "row_count",
        "Tot_Clms_sum",
        "Tot_Drug_Cst_sum",
        "Tot_Benes_sum",
        "Brnd_Name_nunique",
        "Gnrc_Name_nunique",
    ]:
        if col in result.columns:
            result[f"log1p_{col}"] = np.log1p(pd.to_numeric(result[col], errors="coerce").fillna(0.0).clip(lower=0.0))

    trend_cols = [
        c
        for c in [
            "Tot_Clms",
            "Tot_30day_Fills",
            "Tot_Day_Suply",
            "Tot_Drug_Cst",
            "Tot_Benes",
            "GE65_Tot_Clms",
            "GE65_Tot_Drug_Cst",
            "GE65_Tot_Day_Suply",
        ]
        if c in numeric_cols
    ]
    if "Year" in work.columns and trend_cols:
        yearly = work[["Provider", "Year"] + trend_cols].copy()
        yearly["Year"] = pd.to_numeric(yearly["Year"], errors="coerce")
        yearly = yearly.dropna(subset=["Year"]).groupby(["Provider", "Year"], as_index=False)[trend_cols].mean()
        if not yearly.empty:
            span = yearly.groupby("Provider")["Year"].agg(lambda s: float(s.max() - s.min())).reset_index(name="time_span")
            result = result.merge(span, on="Provider", how="left")
            recent = yearly.sort_values("Year").groupby("Provider", as_index=False).tail(1)[["Provider"] + trend_cols]
            recent = recent.rename(columns={c: f"{c}_recent" for c in trend_cols})
            result = result.merge(recent, on="Provider", how="left")
            for col in trend_cols:
                recent_col = f"{col}_recent"
                mean_col = f"{col}_mean"
                sum_col = f"{col}_sum"
                if recent_col in result.columns and mean_col in result.columns:
                    result[f"{col}_recent_to_mean"] = _safe_divide(result[recent_col], result[mean_col])
                if recent_col in result.columns and sum_col in result.columns:
                    result[f"{col}_recent_to_sum"] = _safe_divide(result[recent_col], result[sum_col])

    result = _add_part_d_concentration_features(result, work)
    result = _add_part_d_percentile_rank_features(result)

    if not label_df.empty and {"Provider", "ProviderLabel"}.issubset(label_df.columns):
        labels = label_df[["Provider", "ProviderLabel"]].copy()
        labels["Provider"] = labels["Provider"].astype(str)
        result = result.merge(labels, on="Provider", how="left")
    return result.fillna(0)


def _add_part_d_percentile_rank_features(frame: pd.DataFrame) -> pd.DataFrame:
    rank_features: dict[str, pd.Series] = {}
    for col in frame.columns:
        if col == "Provider" or col.endswith("_mode"):
            continue
        if "risk" in col.lower() or "year" in col.lower() or "source" in col.lower() or "来源文件" in col:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if not values.notna().any() or values.nunique(dropna=True) <= 1:
            continue
        rank_features[f"{col}_pct_rank"] = values.fillna(0.0).rank(method="average", pct=True)
    if not rank_features:
        return frame
    return pd.concat([frame, pd.DataFrame(rank_features, index=frame.index)], axis=1)


def _add_part_d_concentration_features(frame: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    work = claims.copy()
    work["Provider"] = work["Provider"].astype(str)
    for col in ["Tot_Clms", "Tot_Drug_Cst", "Tot_30day_Fills", "Tot_Day_Suply", "Tot_Benes"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    total_agg = {"provider_row_total": ("Provider", "size")}
    total_source_cols = {
        "provider_claim_total": "Tot_Clms",
        "provider_cost_total": "Tot_Drug_Cst",
        "provider_fill_total": "Tot_30day_Fills",
        "provider_day_total": "Tot_Day_Suply",
    }
    for out_col, source_col in total_source_cols.items():
        if source_col in work.columns:
            total_agg[out_col] = (source_col, "sum")
    totals = work.groupby("Provider", as_index=False).agg(**total_agg)

    def merge_ratio_features(features: pd.DataFrame) -> None:
        nonlocal out
        out = out.merge(features, on="Provider", how="left")

    for col, prefix in [("Brnd_Name", "brnd"), ("Gnrc_Name", "gnrc"), ("Prscrbr_Type", "type")]:
        if col not in work.columns:
            continue
        agg_spec = {"cat_rows": (col, "size")}
        for out_col, source_col in [
            ("cat_claims", "Tot_Clms"),
            ("cat_cost", "Tot_Drug_Cst"),
            ("cat_fills", "Tot_30day_Fills"),
            ("cat_days", "Tot_Day_Suply"),
        ]:
            if source_col in work.columns:
                agg_spec[out_col] = (source_col, "sum")
        cat = work.groupby(["Provider", col], sort=False, as_index=False).agg(**agg_spec)
        top = cat.groupby("Provider", as_index=False).max(numeric_only=True).merge(totals, on="Provider", how="left")
        features = top[["Provider"]].copy()
        share_pairs = [
            ("row", "cat_rows", "provider_row_total"),
            ("claim", "cat_claims", "provider_claim_total"),
            ("cost", "cat_cost", "provider_cost_total"),
            ("fill", "cat_fills", "provider_fill_total"),
            ("day", "cat_days", "provider_day_total"),
        ]
        for name, numerator, denominator in share_pairs:
            if numerator in top.columns and denominator in top.columns:
                features[f"{prefix}_top_{name}_share"] = _safe_divide(top[numerator], top[denominator])

        hhi = cat[["Provider"]].copy()
        cat_with_totals = cat.merge(totals, on="Provider", how="left")
        if "cat_claims" in cat_with_totals.columns and "provider_claim_total" in cat_with_totals.columns:
            hhi[f"{prefix}_claims_hhi"] = _safe_divide(cat_with_totals["cat_claims"], cat_with_totals["provider_claim_total"]) ** 2
        if "cat_cost" in cat_with_totals.columns and "provider_cost_total" in cat_with_totals.columns:
            hhi[f"{prefix}_cost_hhi"] = _safe_divide(cat_with_totals["cat_cost"], cat_with_totals["provider_cost_total"]) ** 2
        hhi_cols = [c for c in hhi.columns if c != "Provider"]
        if hhi_cols:
            hhi = hhi.groupby("Provider", as_index=False)[hhi_cols].sum()
            features = features.merge(hhi, on="Provider", how="left")
        merge_ratio_features(features)

    row_features = work[["Provider"]].copy()
    if {"Tot_Drug_Cst", "Tot_Clms"}.issubset(work.columns):
        row_features["row_cost_per_claim"] = _safe_divide(work["Tot_Drug_Cst"], work["Tot_Clms"])
    if {"Tot_Clms", "Tot_Benes"}.issubset(work.columns):
        row_features["row_claims_per_bene"] = _safe_divide(work["Tot_Clms"], work["Tot_Benes"])
    value_cols = [c for c in row_features.columns if c != "Provider"]
    if value_cols:
        row_stats = row_features.groupby("Provider", as_index=False)[value_cols].agg(["mean", "max"])
        row_stats.columns = ["Provider"] + [f"{col}_{stat}" for col, stat in row_stats.columns.tolist()[1:]]
        merge_ratio_features(row_stats)

    for col in ["GE65_Sprsn_Flag", "GE65_Bene_Sprsn_Flag"]:
        if col not in work.columns:
            continue
        values = work[col].astype(str).str.strip()
        flags = pd.DataFrame(
            {
                "Provider": work["Provider"],
                f"{col}_present_share": ((values != "") & (values.str.lower() != "nan")).astype(float),
                f"{col}_suppressed_share": values.isin(["#", "*"]).astype(float),
            }
        )
        merge_ratio_features(flags.groupby("Provider", as_index=False).mean(numeric_only=True))

    return out.fillna(0)


def _smoothed_risk_maps(frame: pd.DataFrame, cols: list[str], global_rate: float, smoothing: float = 20.0) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    for col in cols:
        if col not in frame.columns:
            continue
        grouped = frame.dropna(subset=[col]).copy()
        grouped[col] = grouped[col].astype(str).str.strip()
        grouped = grouped[grouped[col] != ""]
        if grouped.empty:
            continue
        stats = grouped.groupby(col)["ProviderLabel"].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_rate * smoothing) / (stats["count"] + smoothing)
        maps[col] = smoothed.to_dict()
    return maps


def _build_risk_maps(train_claims: pd.DataFrame, train_labels: pd.DataFrame, cols: list[str]) -> tuple[dict[str, dict[str, float]], float]:
    if train_claims.empty or train_labels.empty or "ProviderLabel" not in train_labels.columns:
        return {}, 0.0
    labels = train_labels[["Provider", "ProviderLabel"]].copy()
    labels["Provider"] = labels["Provider"].astype(str)
    labels["ProviderLabel"] = pd.to_numeric(labels["ProviderLabel"], errors="coerce").fillna(0).astype(float)
    frame = train_claims.copy()
    frame["Provider"] = frame["Provider"].astype(str)
    frame = frame.merge(labels, on="Provider", how="left")
    global_rate = float(labels["ProviderLabel"].mean()) if len(labels) else 0.0
    return _smoothed_risk_maps(frame, cols, global_rate), global_rate


def _apply_risk_maps(claims: pd.DataFrame, risk_maps: dict[str, dict[str, float]], global_rate: float) -> pd.DataFrame:
    if claims.empty or not risk_maps:
        return claims
    out = claims.copy()
    for col, mapping in risk_maps.items():
        if col not in out.columns:
            continue
        out[f"{col}_train_risk_rate"] = out[col].astype(str).str.strip().map(mapping).fillna(global_rate).astype(float)
    return out


def _risk_rate_columns(claims: pd.DataFrame) -> list[str]:
    return [c for c in claims.columns if c.endswith("_train_risk_rate")]


def _slope(values: pd.Series, years: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    x = pd.to_numeric(years, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if len(y) < 2 or np.isclose(x.max(), x.min()):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def _mode_or_missing(values: pd.Series) -> str:
    cleaned = values.astype(object).where(pd.notna(values), "__MISSING__").astype(str).str.strip()
    cleaned = cleaned.mask(cleaned.eq(""), "__MISSING__")
    mode = cleaned.mode(dropna=False)
    return "__MISSING__" if len(mode) == 0 else str(mode.iloc[0])


def _normalize_beneficiary(bene: pd.DataFrame) -> pd.DataFrame:
    if bene.empty:
        return bene
    out = bene.copy()
    for c in ["DOB", "DOD"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    if "RenalDiseaseIndicator" in out.columns:
        renal = out["RenalDiseaseIndicator"].astype(str).str.strip().str.upper()
        out["RenalDiseaseIndicator"] = renal.map(
            {"Y": 1, "1": 1, "YES": 1, "TRUE": 1, "0": 0, "N": 0, "NO": 0, "FALSE": 0}
        ).fillna(0).astype(int)
    for c in [x for x in out.columns if x.startswith("ChronicCond_")]:
        out[c] = out[c].replace({"1": 1, "2": 0, 1: 1, 2: 0}).fillna(0).astype(int)
    if "DOB" in out.columns:
        ref = pd.Timestamp("2009-12-31")
        out["Age"] = ((ref - out["DOB"]).dt.days / 365.25).round(0)
    else:
        out["Age"] = np.nan
    out["IsDeceased"] = out["DOD"].notna().astype(int) if "DOD" in out.columns else 0
    return out


def _normalize_claims(df: pd.DataFrame, claim_type: str, time_freq: str = "M") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    out["ClaimType"] = claim_type
    for c in ["InscClaimAmtReimbursed", "DeductibleAmtPaid"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        else:
            out[c] = 0.0
    if "ClaimStartDt" in out.columns:
        out["TimeWindow"] = "TW_" + out["ClaimStartDt"].dt.to_period(time_freq).astype(str).str.replace("-", "_", regex=False)
    else:
        out["TimeWindow"] = "TW_UNKNOWN"
    diag = [c for c in out.columns if c.startswith("ClmDiagnosisCode_")]
    proc = [c for c in out.columns if c.startswith("ClmProcedureCode_")]
    physician_cols = [c for c in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"] if c in out.columns]
    out["DiagnosisCodeCount"] = out[diag].notna().sum(axis=1) if diag else 0
    out["ProcedureCodeCount"] = out[proc].notna().sum(axis=1) if proc else 0
    out = _add_binary_presence_features(out, physician_cols, "Physician")
    if "AdmissionDt" in out.columns and "DischargeDt" in out.columns:
        out["AdmissionDurationDays"] = ((out["DischargeDt"] - out["AdmissionDt"]).dt.days.fillna(0).clip(lower=0) + 1)
    else:
        out["AdmissionDurationDays"] = 0
    out["HasOperatingPhysician"] = out.get("OperatingPhysician", "").astype(str).str.strip().ne("").astype(int) if "OperatingPhysician" in out.columns else 0
    out["HasOtherPhysician"] = out.get("OtherPhysician", "").astype(str).str.strip().ne("").astype(int) if "OtherPhysician" in out.columns else 0
    out["HasAdmitDiagnosis"] = out.get("ClmAdmitDiagnosisCode", "").astype(str).str.strip().ne("").astype(int) if "ClmAdmitDiagnosisCode" in out.columns else 0
    if "ClaimStartDt" in out.columns and "ClaimEndDt" in out.columns:
        out["ClaimDurationDays"] = ((out["ClaimEndDt"] - out["ClaimStartDt"]).dt.days.fillna(0).clip(lower=0) + 1)
    else:
        out["ClaimDurationDays"] = 1
    out["DailyReimbursedAmt"] = out["InscClaimAmtReimbursed"] / out["ClaimDurationDays"].replace(0, 1)
    return out


def _collab_edges(claims: pd.DataFrame, overlap_threshold: int = 2) -> pd.DataFrame:
    columns = [
        "TimeWindow",
        "Provider_src",
        "Provider_dst",
        "shared_bene_count",
        "shared_claim_count",
        "patient_overlap_ratio",
        "collaboration_strength",
        "avg_reimbursed_src",
        "avg_reimbursed_dst",
    ]
    if claims.empty or not {"Provider", "BeneID"}.issubset(claims.columns):
        return pd.DataFrame(columns=columns)

    pb = claims.groupby(["TimeWindow", "Provider", "BeneID"], as_index=False).agg(claim_count=("ClaimID", "nunique"), total_reimbursed=("InscClaimAmtReimbursed", "sum"))
    if pb.empty:
        return pd.DataFrame(columns=columns)

    pb["Provider"] = pb["Provider"].astype(str)
    provider_bene_counts = pb.groupby(["TimeWindow", "Provider"])["BeneID"].nunique()
    pairs = pb.merge(pb, on=["TimeWindow", "BeneID"], suffixes=("_src", "_dst"))
    pairs = pairs[pairs["Provider_src"] < pairs["Provider_dst"]]
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    edges = (
        pairs.groupby(["TimeWindow", "Provider_src", "Provider_dst"], as_index=False)
        .agg(
            shared_bene_count=("BeneID", "nunique"),
            shared_claim_count=("claim_count_src", "sum"),
            sum_reimbursed_src=("total_reimbursed_src", "sum"),
            sum_reimbursed_dst=("total_reimbursed_dst", "sum"),
        )
    )
    edges = edges[edges["shared_bene_count"] >= overlap_threshold].copy()
    if edges.empty:
        return pd.DataFrame(columns=columns)

    def _provider_bene_count(row: pd.Series, provider_col: str) -> int:
        return int(provider_bene_counts.get((row["TimeWindow"], row[provider_col]), 0))

    src_counts = edges.apply(lambda r: _provider_bene_count(r, "Provider_src"), axis=1)
    dst_counts = edges.apply(lambda r: _provider_bene_count(r, "Provider_dst"), axis=1)
    union_counts = (src_counts + dst_counts - edges["shared_bene_count"]).replace(0, np.nan)
    edges["patient_overlap_ratio"] = (edges["shared_bene_count"] / union_counts).fillna(0.0)
    edges["collaboration_strength"] = edges["shared_bene_count"].astype(float) * (1.0 + edges["patient_overlap_ratio"])
    edges["avg_reimbursed_src"] = _safe_divide(edges["sum_reimbursed_src"], edges["shared_bene_count"])
    edges["avg_reimbursed_dst"] = _safe_divide(edges["sum_reimbursed_dst"], edges["shared_bene_count"])
    return edges[columns].reset_index(drop=True)


def _static_provider_features(claims: pd.DataFrame, bene: pd.DataFrame, labels: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if claims.empty and labels.empty:
        return pd.DataFrame()
    if claims.empty:
        base = labels[["Provider"]].copy()
        base["total_claims"] = 0
        base["total_beneficiaries"] = 0
        base["total_reimbursed"] = 0.0
        base["avg_reimbursed"] = 0.0
        base["avg_deductible"] = 0.0
        base["inpatient_claims"] = 0
        base["outpatient_claims"] = 0
        base["avg_claim_duration"] = 0.0
        base["avg_diag_code_count"] = 0.0
        base["avg_proc_code_count"] = 0.0
        base["avg_patient_age"] = 0.0
        base["deceased_patient_ratio"] = 0.0
        base["renal_patient_ratio"] = 0.0
        base["collab_provider_count"] = 0
        base["shared_patient_volume"] = 0
        base["avg_overlap_ratio"] = 0.0
        base["avg_collaboration_strength"] = 0.0
        return base.merge(labels, on="Provider", how="left")

    chronic_cols = [c for c in bene.columns if c.startswith("ChronicCond_")]
    agg = claims.groupby("Provider", as_index=False).agg(
        total_claims=("ClaimID", "nunique"),
        total_beneficiaries=("BeneID", "nunique"),
        total_reimbursed=("InscClaimAmtReimbursed", "sum"),
        avg_reimbursed=("InscClaimAmtReimbursed", "mean"),
        avg_deductible=("DeductibleAmtPaid", "mean"),
        inpatient_claims=("ClaimType", lambda s: int((s == "inpatient").sum())),
        outpatient_claims=("ClaimType", lambda s: int((s == "outpatient").sum())),
        avg_claim_duration=("ClaimDurationDays", "mean"),
        avg_diag_code_count=("DiagnosisCodeCount", "mean"),
        avg_proc_code_count=("ProcedureCodeCount", "mean"),
        avg_patient_age=("Age", "mean"),
        deceased_patient_ratio=("IsDeceased", "mean"),
        renal_patient_ratio=("RenalDiseaseIndicator", "mean"),
    )
    amount_cols = [c for c in ["InscClaimAmtReimbursed", "DailyReimbursedAmt", "DeductibleAmtPaid"] if c in claims.columns]
    if amount_cols:
        amount_frame = claims[["Provider"] + amount_cols].copy()
        for col in amount_cols:
            amount_frame[col] = pd.to_numeric(amount_frame[col], errors="coerce").fillna(0.0)
        amount_parts = []
        for col in amount_cols:
            stats = amount_frame.groupby("Provider")[col].agg(
                [
                    ("median", "median"),
                    ("std", "std"),
                    ("max", "max"),
                    ("p90", lambda s: float(s.quantile(0.90))),
                    ("p95", lambda s: float(s.quantile(0.95))),
                ]
            )
            stats.columns = [f"{col}_{name}" for name in stats.columns]
            mean = amount_frame.groupby("Provider")[col].mean().replace(0, np.nan)
            median = amount_frame.groupby("Provider")[col].median().replace(0, np.nan)
            stats[f"{col}_cv"] = (stats[f"{col}_std"] / mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            stats[f"{col}_max_to_mean"] = (stats[f"{col}_max"] / mean).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            stats[f"{col}_p95_to_median"] = (stats[f"{col}_p95"] / median).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            global_p90 = float(amount_frame[col].quantile(0.90))
            global_p95 = float(amount_frame[col].quantile(0.95))
            flags = amount_frame.assign(
                _is_zero=(amount_frame[col] <= 0).astype(float),
                _is_global_p90=(amount_frame[col] >= global_p90).astype(float),
                _is_global_p95=(amount_frame[col] >= global_p95).astype(float),
            )
            flag_stats = flags.groupby("Provider")[["_is_zero", "_is_global_p90", "_is_global_p95"]].mean()
            flag_stats.columns = [
                f"{col}_zero_share",
                f"{col}_global_p90_share",
                f"{col}_global_p95_share",
            ]
            amount_parts.append(pd.concat([stats, flag_stats], axis=1).reset_index())
        for part in amount_parts:
            agg = agg.merge(part.fillna(0), on="Provider", how="left")

    if {"Provider", "BeneID", "ClaimID"}.issubset(claims.columns):
        bene_claims = claims.groupby(["Provider", "BeneID"])["ClaimID"].nunique().reset_index(name="bene_claim_count")
        repeat_stats = bene_claims.groupby("Provider", as_index=False).agg(
            claims_per_beneficiary_mean=("bene_claim_count", "mean"),
            claims_per_beneficiary_max=("bene_claim_count", "max"),
            repeat_beneficiary_share=("bene_claim_count", lambda s: float((s > 1).mean())),
        )
        agg = agg.merge(repeat_stats.fillna(0), on="Provider", how="left")

    physician_cols = [c for c in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"] if c in claims.columns]
    if physician_cols:
        physician_rows = []
        for provider, group in claims.groupby("Provider", sort=False):
            values = group[physician_cols].stack().astype(str).str.strip()
            values = values[(values != "") & (values.str.lower() != "nan") & (values != "0")]
            counts = values.value_counts()
            total = float(counts.sum())
            physician_rows.append(
                {
                    "Provider": provider,
                    "physician_all_nunique": int(counts.size),
                    "physician_all_top_share": float(counts.iloc[0] / total) if total else 0.0,
                    "physician_all_entropy": _weighted_entropy(counts) if total else 0.0,
                }
            )
        agg = agg.merge(pd.DataFrame(physician_rows), on="Provider", how="left")

    relative_keys = [c for c in ["ClaimType", "State", "Rfrg_Prvdr_State_Abrvtn", "Rfrg_Prvdr_Spclty_Cd", "RBCS_Id"] if c in claims.columns]
    relative_amount_cols = [c for c in ["InscClaimAmtReimbursed", "DailyReimbursedAmt"] if c in claims.columns]
    for key in relative_keys:
        rel_frame = claims[["Provider", key] + relative_amount_cols].copy()
        if rel_frame.empty or not relative_amount_cols:
            continue
        rel_rows = []
        for col in relative_amount_cols:
            rel_frame[col] = pd.to_numeric(rel_frame[col], errors="coerce").fillna(0.0)
            key_mean = rel_frame.groupby(key)[col].transform("mean")
            key_std = rel_frame.groupby(key)[col].transform("std").replace(0, np.nan)
            z_col = f"_{key}_{col}_z"
            rel_frame[z_col] = ((rel_frame[col] - key_mean) / key_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            rel_rows.append(z_col)
        rel_stats = rel_frame.groupby("Provider")[rel_rows].agg(["mean", "max"]).reset_index()
        rel_stats.columns = ["Provider"] + [
            f"{col[0].strip('_')}_{col[1]}" for col in rel_stats.columns.tolist()[1:]
        ]
        agg = agg.merge(rel_stats.fillna(0), on="Provider", how="left")

    if chronic_cols:
        chronic_means = claims.groupby("Provider", as_index=False)[chronic_cols].mean()
        agg = agg.merge(chronic_means, on="Provider", how="left")

    if edges.empty:
        rel = pd.DataFrame({"Provider": agg["Provider"], "collab_provider_count": 0, "shared_patient_volume": 0, "avg_overlap_ratio": 0.0, "avg_collaboration_strength": 0.0})
    else:
        weight_col = "collaboration_strength" if "collaboration_strength" in edges.columns else "shared_bene_count"
        overlap_col = "patient_overlap_ratio" if "patient_overlap_ratio" in edges.columns else "shared_bene_count"
        a = edges[["Provider_src", "Provider_dst", "shared_bene_count", weight_col, overlap_col]].rename(
            columns={"Provider_src": "Provider", "Provider_dst": "Neighbor", weight_col: "edge_weight", overlap_col: "edge_overlap"}
        )
        b = edges[["Provider_dst", "Provider_src", "shared_bene_count", weight_col, overlap_col]].rename(
            columns={"Provider_dst": "Provider", "Provider_src": "Neighbor", weight_col: "edge_weight", overlap_col: "edge_overlap"}
        )
        rel = (
            pd.concat([a, b], ignore_index=True)
            .groupby("Provider", as_index=False)
            .agg(
                collab_provider_count=("Neighbor", "nunique"),
                shared_patient_volume=("shared_bene_count", "sum"),
                avg_overlap_ratio=("edge_overlap", "mean"),
                max_overlap_ratio=("edge_overlap", "max"),
                avg_collaboration_strength=("edge_weight", "mean"),
                max_collaboration_strength=("edge_weight", "max"),
            )
        )

    extra_numeric = [
        c
        for c in [
            "Gender",
            "Race",
            "State",
            "County",
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
            "Physician_count",
            "AdmissionDurationDays",
            "HasOperatingPhysician",
            "HasOtherPhysician",
            "HasAdmitDiagnosis",
            "Tot_Suplrs",
            "Tot_Suplr_Benes",
            "Tot_Suplr_Clms",
            "Tot_Suplr_Srvcs",
            "Avg_Suplr_Sbmtd_Chrg",
            "Avg_Suplr_Mdcr_Alowd_Amt",
            "Avg_Suplr_Mdcr_Pymt_Amt",
            "Avg_Suplr_Mdcr_Stdzd_Amt",
            "Year",
            "ratio_submitted_to_allowed",
            "ratio_payment_to_allowed",
            "ratio_std_to_payment",
            "ratio_services_to_claims",
            "ratio_benes_to_claims",
            "HCPCS_CD_train_risk_rate",
            "RBCS_Id_train_risk_rate",
            "Rfrg_Prvdr_Spclty_Cd_train_risk_rate",
            "Rfrg_Prvdr_Spclty_Desc_train_risk_rate",
        ]
        if c in claims.columns
    ]
    extra_numeric = list(dict.fromkeys(extra_numeric + _risk_rate_columns(claims)))
    if extra_numeric:
        numeric_frame = claims[["Provider"] + extra_numeric].copy()
        for col in extra_numeric:
            numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors="coerce")
        extra = numeric_frame.groupby("Provider", as_index=False).agg(["mean", "sum", "std"])
        extra.columns = ["Provider"] + [f"{col}_{stat}" for col, stat in extra.columns.tolist()[1:]]
        agg = agg.merge(extra.fillna(0), on="Provider", how="left")

    if {"IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"}.issubset(claims.columns):
        claims["beneficiary_total_annual_reimbursement"] = (
            pd.to_numeric(claims["IPAnnualReimbursementAmt"], errors="coerce").fillna(0.0)
            + pd.to_numeric(claims["OPAnnualReimbursementAmt"], errors="coerce").fillna(0.0)
        )
        bene_cost = claims.groupby("Provider", as_index=False).agg(
            beneficiary_total_annual_reimbursement_mean=("beneficiary_total_annual_reimbursement", "mean"),
            beneficiary_total_annual_reimbursement_sum=("beneficiary_total_annual_reimbursement", "sum"),
            beneficiary_total_annual_reimbursement_std=("beneficiary_total_annual_reimbursement", "std"),
        )
        agg = agg.merge(bene_cost.fillna(0), on="Provider", how="left")

    categorical_cols = [
        c
        for c in [
            "Rfrg_Prvdr_State_Abrvtn",
            "Rfrg_Prvdr_Spclty_Cd",
            "Rfrg_Prvdr_Spclty_Desc",
            "RBCS_Lvl",
            "RBCS_Id",
            "HCPCS_CD",
            "Suplr_Rentl_Ind",
            "ClaimType",
            "Gender",
            "Race",
            "State",
            "County",
            "AttendingPhysician",
            "OperatingPhysician",
            "OtherPhysician",
            "ClmAdmitDiagnosisCode",
            "DiagnosisGroupCode",
        ]
        if c in claims.columns
    ]
    if categorical_cols:
        cat_rows = []
        for provider, group in claims.groupby("Provider", sort=False):
            row = {"Provider": provider}
            n = max(1, len(group))
            for col in categorical_cols:
                values = group[col].astype(str).str.strip()
                counts = values[values != ""].value_counts()
                row[f"{col}_nunique"] = int(counts.size)
                row[f"{col}_top_share"] = float(counts.iloc[0] / n) if not counts.empty else 0.0
                row[f"{col}_entropy"] = _weighted_entropy(counts) if not counts.empty else 0.0
            cat_rows.append(row)
        agg = agg.merge(pd.DataFrame(cat_rows), on="Provider", how="left")

    code_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_") or c.startswith("ClmProcedureCode_")]
    if code_cols:
        code_rows = []
        for provider, group in claims.groupby("Provider", sort=False):
            row = {"Provider": provider}
            for prefix, cols in [
                ("diag", [c for c in code_cols if c.startswith("ClmDiagnosisCode_")]),
                ("proc", [c for c in code_cols if c.startswith("ClmProcedureCode_")]),
            ]:
                values = group[cols].stack().astype(str).str.strip() if cols else pd.Series(dtype=str)
                counts = values[(values != "") & (values.str.lower() != "nan")].value_counts()
                row[f"{prefix}_code_nunique"] = int(counts.size)
                row[f"{prefix}_code_entropy"] = _weighted_entropy(counts) if not counts.empty else 0.0
                row[f"{prefix}_code_top_share"] = float(counts.iloc[0] / counts.sum()) if not counts.empty and counts.sum() else 0.0
            code_rows.append(row)
        agg = agg.merge(pd.DataFrame(code_rows), on="Provider", how="left")

    if "Year" in claims.columns:
        trend_metrics = [
            c
            for c in [
                "InscClaimAmtReimbursed",
                "Tot_Suplr_Clms",
                "Tot_Suplr_Benes",
                "Tot_Suplr_Srvcs",
                "Avg_Suplr_Sbmtd_Chrg",
                "Avg_Suplr_Mdcr_Alowd_Amt",
                "Avg_Suplr_Mdcr_Pymt_Amt",
                "Avg_Suplr_Mdcr_Stdzd_Amt",
            ]
            if c in claims.columns
        ]
        if trend_metrics:
            trend_frame = claims[["Provider", "Year"] + trend_metrics].copy()
            trend_frame["Year"] = pd.to_numeric(trend_frame["Year"], errors="coerce")
            for col in trend_metrics:
                trend_frame[col] = pd.to_numeric(trend_frame[col], errors="coerce").fillna(0.0)
            yearly = trend_frame.groupby(["Provider", "Year"], as_index=False)[trend_metrics].mean()
            rows = []
            for provider, group in yearly.groupby("Provider", sort=False):
                row = {"Provider": provider}
                years = group["Year"]
                recent_year = years.max()
                recent = group[group["Year"] == recent_year]
                for col in trend_metrics:
                    row[f"{col}_year_slope"] = _slope(group[col], years)
                    row[f"{col}_year_range"] = float(group[col].max() - group[col].min())
                    row[f"{col}_recent_mean"] = float(recent[col].mean()) if not recent.empty else 0.0
                rows.append(row)
            agg = agg.merge(pd.DataFrame(rows), on="Provider", how="left")

    out = agg.merge(rel, on="Provider", how="left")
    return out.fillna(0)


def _temporal_provider_features(claims: pd.DataFrame, bene: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame(columns=["Provider", "TimeWindow"])
    out = claims.groupby(["Provider", "TimeWindow"], as_index=False).agg(
        inpatient_cnt=("ClaimType", lambda s: int((s == "inpatient").sum())),
        outpatient_cnt=("ClaimType", lambda s: int((s == "outpatient").sum())),
        claim_cnt=("ClaimID", "nunique"),
        bene_cnt=("BeneID", "nunique"),
        amount_sum=("InscClaimAmtReimbursed", "sum"),
        amount_mean=("InscClaimAmtReimbursed", "mean"),
        amount_std=("InscClaimAmtReimbursed", "std"),
        deductible_sum=("DeductibleAmtPaid", "sum"),
        duration_mean=("ClaimDurationDays", "mean"),
        diag_mean=("DiagnosisCodeCount", "mean"),
        proc_mean=("ProcedureCodeCount", "mean"),
        age_mean=("Age", "mean"),
        deceased_ratio=("IsDeceased", "mean"),
        renal_ratio=("RenalDiseaseIndicator", "mean"),
        daily_reimbursed_mean=("DailyReimbursedAmt", "mean"),
        physician_count_mean=("Physician_count", "mean"),
        has_operating_physician_ratio=("HasOperatingPhysician", "mean"),
        has_other_physician_ratio=("HasOtherPhysician", "mean"),
        has_admit_diagnosis_ratio=("HasAdmitDiagnosis", "mean"),
        admission_duration_mean=("AdmissionDurationDays", "mean"),
    )
    temporal_numeric = [
        c
        for c in [
            "NoOfMonths_PartACov",
            "NoOfMonths_PartBCov",
            "IPAnnualReimbursementAmt",
            "IPAnnualDeductibleAmt",
            "OPAnnualReimbursementAmt",
            "OPAnnualDeductibleAmt",
        ]
        if c in claims.columns
    ]
    temporal_numeric = list(dict.fromkeys(temporal_numeric + _risk_rate_columns(claims)))
    if temporal_numeric:
        numeric = claims[["Provider", "TimeWindow"] + temporal_numeric].copy()
        for col in temporal_numeric:
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce").fillna(0.0)
        extra = numeric.groupby(["Provider", "TimeWindow"], as_index=False).agg(["mean", "sum", "std"])
        extra.columns = ["Provider", "TimeWindow"] + [f"{col}_{stat}" for col, stat in extra.columns.tolist()[2:]]
        out = out.merge(extra.fillna(0), on=["Provider", "TimeWindow"], how="left")
    return out.fillna(0)


def preprocess_medical_insurance_data(
    dataset_root: str | Path,
    output_root: str | Path,
    time_freq: str = "28D",
    overlap_threshold: int = PREPROCESSING["graph_overlap_threshold"],
    write_claims_enriched: bool = False,
) -> dict:
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    result: dict[str, object] = {}
    prepared: dict[str, dict[str, pd.DataFrame]] = {}

    for split_index, split in enumerate(["Train", "Test"], start=1):
        log_progress("预处理 split", split_index, 2, extra=split)
        csvs = _split_source_csvs(dataset_root, split)
        if not csvs:
            log_line(f"{split} 未找到 CSV，跳过", tag="WARN")
            continue

        with log_stage(f"{split} 读取与清洗原始数据"):
            is_dmepos_flat = len(csvs) == 1 and _is_dmepos_flat_file(csvs[0])
            if is_dmepos_flat:
                raw = _safe_read_csv(csvs[0])
                claims, label_df = _prepare_dmepos_flat(raw, keep_label_in_claims=True)
                claims = _add_dmepos_ratio_columns(claims)
                bene = pd.DataFrame()
            else:
                provider_file = _find_provider_file(csvs, split)
                label_df = _safe_read_csv(provider_file) if provider_file else pd.DataFrame()
                bene_file = _find_first(csvs, ["beneficiary"])
                ip_file = _find_first(csvs, ["inpatient"])
                op_file = _find_first(csvs, ["outpatient"])
                bene = _normalize_beneficiary(_safe_read_csv(bene_file)) if bene_file else pd.DataFrame()
                ip = _normalize_claims(_safe_read_csv(ip_file), "inpatient", time_freq) if ip_file else pd.DataFrame()
                op = _normalize_claims(_safe_read_csv(op_file), "outpatient", time_freq) if op_file else pd.DataFrame()
                claims = pd.concat([ip, op], ignore_index=True, sort=False)

        if label_df.empty:
            label_df = pd.DataFrame(columns=["Provider", "ProviderLabel"])
        else:
            if "PotentialFraud" in label_df.columns:
                label_df = label_df.rename(columns={"PotentialFraud": "ProviderLabel"})
            if "ProviderFinalLabel" in label_df.columns and "ProviderLabel" not in label_df.columns:
                label_df = label_df.rename(columns={"ProviderFinalLabel": "ProviderLabel"})
            if "ProviderLabel" in label_df.columns:
                label_df["ProviderLabel"] = _normalize_label_series(label_df["ProviderLabel"])
            if "ProviderLabel" not in label_df.columns:
                label_df["ProviderLabel"] = np.nan
            label_df = label_df[[c for c in ["Provider", "ProviderLabel"] if c in label_df.columns]].drop_duplicates()
            if "Provider" in label_df.columns:
                label_df["Provider"] = label_df["Provider"].astype(str)

        prepared[split.lower()] = {"claims": claims, "bene": bene, "labels": label_df, "is_dmepos": is_dmepos_flat}

    train_claims_for_risk = prepared.get("train", {}).get("claims", pd.DataFrame())
    if prepared.get("train", {}).get("is_dmepos", False):
        dmepos_categorical_cols = _dmepos_categorical_columns(train_claims_for_risk)
        risk_maps, global_risk_rate = _dmepos_risk_maps(train_claims_for_risk, dmepos_categorical_cols)
        frequency_maps = _dmepos_frequency_maps(train_claims_for_risk, dmepos_categorical_cols)
    else:
        risk_cols = [
            "HCPCS_CD",
            "RBCS_Id",
            "Rfrg_Prvdr_Spclty_Cd",
            "Rfrg_Prvdr_Spclty_Desc",
        ]
        risk_cols = [c for c in dict.fromkeys(risk_cols) if c in train_claims_for_risk.columns]
        risk_maps, global_risk_rate = _build_risk_maps(
            train_claims_for_risk,
            prepared.get("train", {}).get("labels", pd.DataFrame()),
            risk_cols,
        )
        frequency_maps = {}

    for split_index, split in enumerate(["Train", "Test"], start=1):
        split_key = split.lower()
        if split_key not in prepared:
            continue
        log_progress("构建 split 特征", split_index, 2, extra=split)
        claims = prepared[split_key]["claims"]
        bene = prepared[split_key]["bene"]
        label_df = prepared[split_key]["labels"]
        with log_stage(f"{split} 构建 provider 特征"):
            if prepared[split_key].get("is_dmepos", False):
                edges = pd.DataFrame(columns=["TimeWindow", "Provider_src", "Provider_dst", "shared_bene_count", "shared_claim_count", "patient_overlap_ratio", "collaboration_strength", "avg_reimbursed_src", "avg_reimbursed_dst"])
                static_features = _dmepos_adaptive_provider_features(
                    claims,
                    label_df,
                    risk_maps,
                    global_risk_rate,
                    frequency_maps,
                    split_key=split_key,
                )
            else:
                claims = _apply_risk_maps(claims, risk_maps, global_risk_rate)
                if not claims.empty and not bene.empty and "BeneID" in claims.columns:
                    claims = claims.merge(bene, on="BeneID", how="left")
                elif not claims.empty:
                    claims["Age"] = claims.get("Age", np.nan)
                    claims["IsDeceased"] = claims.get("IsDeceased", 0)
                    claims["RenalDiseaseIndicator"] = claims.get("RenalDiseaseIndicator", 0)

                edges = _collab_edges(claims, overlap_threshold=overlap_threshold)
                static_features = _static_provider_features(claims, bene, label_df, edges)
            temporal_features = _temporal_provider_features(claims, bene, label_df)

        with log_stage(f"{split} 写出预处理文件"):
            out_dir = output_root / split.lower()
            out_dir.mkdir(parents=True, exist_ok=True)
            static_features.to_csv(out_dir / "provider_static_features.csv", index=False)
            temporal_features.to_csv(out_dir / "provider_temporal_features.csv", index=False)
            edges.to_csv(out_dir / "provider_graph_features.csv", index=False)
            label_df.to_csv(out_dir / "provider_labels.csv", index=False)
            if write_claims_enriched:
                claims.to_csv(out_dir / "claims_enriched.csv", index=False)
            else:
                stale_claims = out_dir / "claims_enriched.csv"
                if stale_claims.exists():
                    stale_claims.unlink()

        print_table(
            f"{split} 预处理摘要",
            [
                {
                    "providers": len(label_df),
                    "claims": len(claims),
                    "static_rows": len(static_features),
                    "temporal_rows": len(temporal_features),
                    "graph_edges": len(edges),
                }
            ],
        )

        result[split.lower()] = {
            "provider_labels": label_df,
            "provider_static_features": static_features,
            "provider_temporal_features": temporal_features,
            "provider_graph_features": edges,
        }

    return result


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    preprocess_medical_insurance_data(base / "01_Dataset", base / "05_Outputs" / "manual_preprocess" / "02_Data_Preprocessing")
    print("预处理完成")
