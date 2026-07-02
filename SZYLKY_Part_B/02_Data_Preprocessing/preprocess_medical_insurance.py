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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    direct_candidates = [
        dataset_root / f"DMEPOS_{split}.csv",
        dataset_root / f"Part_B_{split}.csv",
        dataset_root / f"part_b_{split.lower()}.csv",
    ]
    for direct in direct_candidates:
        if direct.exists():
            return [direct]
    split_lower = split.lower()
    return sorted(
        [
            path
            for path in dataset_root.glob("*.csv")
            if split_lower in path.stem.lower() and not path.name.startswith("._")
        ]
    )


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
            "否": 0,
            "no": 0,
            "n": 0,
            "false": 0,
            "0": 0,
            "非诈骗": 0,
        }
    )
    return pd.to_numeric(mapped.where(mapped.notna(), label_series), errors="coerce")


def _is_dmepos_flat(df: pd.DataFrame) -> bool:
    provider_cols = {"Rfrg_NPI", "Rndrng_NPI"}
    return bool(provider_cols.intersection(df.columns)) and "Year" in df.columns


def _is_dmepos_flat_file(path: Path) -> bool:
    return _is_dmepos_flat(_safe_read_csv(path, nrows=0))


def _standardize_dmepos_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Rndrng_NPI": "Rfrg_NPI",
        "Rndrng_Prvdr_Crdntls": "Rfrg_Prvdr_Crdntls",
        "Rndrng_Prvdr_Ent_Cd": "Rfrg_Prvdr_Ent_Cd",
        "Rndrng_Prvdr_City": "Rfrg_Prvdr_City",
        "Rndrng_Prvdr_State_FIPS": "Rfrg_Prvdr_State_FIPS",
        "Rndrng_Prvdr_State_Abrvtn": "Rfrg_Prvdr_State_Abrvtn",
        "Rndrng_Prvdr_Zip5": "Rfrg_Prvdr_Zip5",
        "Rndrng_Prvdr_RUCA": "Rfrg_Prvdr_RUCA",
        "Rndrng_Prvdr_RUCA_Desc": "Rfrg_Prvdr_RUCA_Desc",
        "Rndrng_Prvdr_Cntry": "Rfrg_Prvdr_Cntry",
        "Rndrng_Prvdr_Type": "Rfrg_Prvdr_Spclty_Desc",
        "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": "Rfrg_Prvdr_Mdcr_Prtcptg_Ind",
        "HCPCS_Cd": "HCPCS_CD",
        "Tot_Benes": "Tot_Suplr_Benes",
        "Tot_Srvcs": "Tot_Suplr_Srvcs",
        "Tot_Bene_Day_Srvcs": "Tot_Suplr_Clms",
        "Avg_Sbmtd_Chrg": "Avg_Suplr_Sbmtd_Chrg",
        "Avg_Mdcr_Alowd_Amt": "Avg_Suplr_Mdcr_Alowd_Amt",
        "Avg_Mdcr_Pymt_Amt": "Avg_Suplr_Mdcr_Pymt_Amt",
        "Avg_Mdcr_Stdzd_Amt": "Avg_Suplr_Mdcr_Stdzd_Amt",
        "来源文件": "source_file",
        "是否诈骗": "ProviderLabel",
        "是否欺诈": "ProviderLabel",
    }
    return df.rename(columns={src: dst for src, dst in rename_map.items() if src in df.columns and dst not in df.columns})


def _prepare_dmepos_flat(df: pd.DataFrame, *, keep_label_in_claims: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["Provider", "ProviderLabel"])

    out = _standardize_dmepos_columns(df.copy())
    out["Provider"] = out["Rfrg_NPI"].astype(str)
    out["ClaimID"] = np.arange(len(out)).astype(str)
    out["BeneID"] = out["Provider"] + "_" + out["ClaimID"]
    out["ClaimType"] = "dmepos"
    out["TimeWindow"] = "TW_" + pd.to_numeric(out["Year"], errors="coerce").fillna(0).astype(int).astype(str)

    amount_col = "Avg_Suplr_Mdcr_Pymt_Amt" if "Avg_Suplr_Mdcr_Pymt_Amt" in out.columns else "Avg_Suplr_Mdcr_Alowd_Amt"
    service_col = "Tot_Suplr_Srvcs" if "Tot_Suplr_Srvcs" in out.columns else "Tot_Suplr_Clms"
    out["_service_count"] = pd.to_numeric(out.get(service_col, 1), errors="coerce").fillna(1).clip(lower=1)
    out["InscClaimAmtReimbursed"] = pd.to_numeric(out.get(amount_col, 0), errors="coerce").fillna(0) * out["_service_count"]
    service_count = pd.to_numeric(out.get("Tot_Suplr_Srvcs", out["_service_count"]), errors="coerce").fillna(0.0)
    claim_count = pd.to_numeric(out.get("Tot_Suplr_Clms", 0), errors="coerce").fillna(0.0)
    bene_count = pd.to_numeric(out.get("Tot_Suplr_Benes", 0), errors="coerce").fillna(0.0)
    submitted = pd.to_numeric(out.get("Avg_Suplr_Sbmtd_Chrg", 0), errors="coerce").fillna(0.0)
    allowed = pd.to_numeric(out.get("Avg_Suplr_Mdcr_Alowd_Amt", 0), errors="coerce").fillna(0.0)
    payment = pd.to_numeric(out.get("Avg_Suplr_Mdcr_Pymt_Amt", 0), errors="coerce").fillna(0.0)
    standardized = pd.to_numeric(out.get("Avg_Suplr_Mdcr_Stdzd_Amt", 0), errors="coerce").fillna(0.0)
    out["submitted_charge_total"] = submitted * service_count
    out["allowed_amount_total"] = allowed * service_count
    out["payment_amount_total"] = payment * service_count
    out["standardized_amount_total"] = standardized * service_count
    out["charge_allowed_gap"] = submitted - allowed
    out["allowed_payment_gap"] = allowed - payment
    out["standardized_payment_gap"] = standardized - payment
    out["service_per_bene"] = _safe_divide(service_count, bene_count)
    out["claim_per_bene"] = _safe_divide(claim_count, bene_count)
    out["payment_per_bene"] = _safe_divide(out["payment_amount_total"], bene_count)
    out["allowed_per_bene"] = _safe_divide(out["allowed_amount_total"], bene_count)
    out["submitted_per_bene"] = _safe_divide(out["submitted_charge_total"], bene_count)
    out["service_per_claim"] = _safe_divide(service_count, claim_count)
    out["DeductibleAmtPaid"] = 0.0
    out["ClaimDurationDays"] = 1.0
    out["DailyReimbursedAmt"] = out["InscClaimAmtReimbursed"]
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


def _provider_categorical_stats(
    work: pd.DataFrame,
    col: str,
    provider_sizes: pd.Series,
    categorical_risk_maps: dict[str, dict[str, float]],
    global_risk: float,
    frequency_map: dict[str, float] | None,
    risk_values: pd.Series | None,
    indicator_levels: list[str] | None = None,
) -> pd.DataFrame:
    values = work[["Provider", col]].copy()
    values[col] = values[col].astype(str).str.strip().replace("", "__MISSING__")
    counts = values.groupby(["Provider", col], sort=False).size().reset_index(name="_count")
    if counts.empty:
        return pd.DataFrame({"Provider": provider_sizes.index.astype(str)})

    top_idx = counts.groupby("Provider", sort=False)["_count"].idxmax()
    mode = counts.loc[top_idx, ["Provider", col, "_count"]].rename(
        columns={col: f"{col}_mode", "_count": f"{col}_top_count"}
    )
    out = provider_sizes.rename("_provider_count").reset_index()
    out.columns = ["Provider", "_provider_count"]
    out = out.merge(mode, on="Provider", how="left")
    out[f"{col}_top_share"] = _safe_divide(out[f"{col}_top_count"], out["_provider_count"])

    nunique = counts.groupby("Provider", sort=False).size().rename(f"{col}_nunique").reset_index()
    out = out.merge(nunique, on="Provider", how="left")
    counts = counts.merge(out[["Provider", "_provider_count"]], on="Provider", how="left")
    counts["_p"] = _safe_divide(counts["_count"], counts["_provider_count"])
    counts["_entropy_term"] = -(counts["_p"] * np.log(counts["_p"].replace(0, np.nan))).fillna(0.0)
    entropy = counts.groupby("Provider", sort=False)["_entropy_term"].sum().rename(f"{col}_entropy").reset_index()
    out = out.merge(entropy, on="Provider", how="left")
    out[f"{col}_normalized_entropy"] = _safe_divide(out[f"{col}_entropy"], np.log(out[f"{col}_nunique"].clip(lower=1)))
    out[f"{col}_nunique_per_row"] = _safe_divide(out[f"{col}_nunique"], out["_provider_count"])

    if frequency_map:
        freq_values = values.copy()
        freq_values["_freq"] = freq_values[col].map(frequency_map).fillna(0.0).astype(float)
        freq_stats = (
            freq_values.groupby("Provider", sort=False)["_freq"]
            .agg(["mean", "max", "min"])
            .rename(columns={"mean": f"{col}_freq_mean", "max": f"{col}_freq_max", "min": f"{col}_freq_min"})
            .reset_index()
        )
        rare_share = (
            freq_values.assign(_rare=(freq_values["_freq"] <= 0.001).astype(float))
            .groupby("Provider", sort=False)["_rare"]
            .mean()
            .rename(f"{col}_rare_share")
            .reset_index()
        )
        out = out.merge(freq_stats, on="Provider", how="left").merge(rare_share, on="Provider", how="left")

    if risk_values is not None or categorical_risk_maps.get(col):
        if risk_values is None:
            risk = values[col].map(categorical_risk_maps.get(col, {})).fillna(global_risk).astype(float)
        else:
            risk = risk_values.reindex(work.index).fillna(global_risk).astype(float)
        risk_stats = (
            pd.DataFrame({"Provider": work["Provider"].to_numpy(), "_risk": risk.to_numpy()})
            .groupby("Provider", sort=False)["_risk"]
            .agg(["mean", "max", "min", "std"])
            .rename(
                columns={
                    "mean": f"{col}_risk_mean",
                    "max": f"{col}_risk_max",
                    "min": f"{col}_risk_min",
                    "std": f"{col}_risk_std",
                }
            )
            .reset_index()
        )
        risk_stats[f"{col}_target_score_mean"] = risk_stats[f"{col}_risk_mean"]
        risk_stats[f"{col}_target_score_max"] = risk_stats[f"{col}_risk_max"]
        risk_stats[f"{col}_target_score_min"] = risk_stats[f"{col}_risk_min"]
        risk_stats[f"{col}_target_score_std"] = risk_stats[f"{col}_risk_std"]
        out = out.merge(risk_stats, on="Provider", how="left")

    for level in indicator_levels or []:
        level = str(level)
        feature_name = f"{col}_share_{_safe_feature_token(level)}"
        hit = (
            values.assign(_hit=(values[col].astype(str) == level).astype(float))
            .groupby("Provider", sort=False)["_hit"]
            .mean()
            .rename(feature_name)
            .reset_index()
        )
        out = out.merge(hit, on="Provider", how="left")

    return out.drop(columns=["_provider_count", f"{col}_top_count"], errors="ignore").fillna(0)


def _safe_feature_token(value: object, max_length: int = 48) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    token = "_".join(part for part in token.split("_") if part)
    return (token or "missing")[:max_length]


def _provider_year_slope(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    grouped = frame[["Provider", "Year", value_col]].dropna(subset=["Year"]).copy()
    if grouped.empty:
        return pd.DataFrame(columns=["Provider", f"{value_col}_slope"])
    grouped[value_col] = pd.to_numeric(grouped[value_col], errors="coerce").fillna(0.0)
    means = grouped.groupby("Provider", sort=False)[["Year", value_col]].transform("mean")
    grouped["_x_centered"] = grouped["Year"] - means["Year"]
    grouped["_y_centered"] = grouped[value_col] - means[value_col]
    grouped["_xy"] = grouped["_x_centered"] * grouped["_y_centered"]
    grouped["_xx"] = grouped["_x_centered"] * grouped["_x_centered"]
    sums = grouped.groupby("Provider", sort=False)[["_xy", "_xx"]].sum().reset_index()
    sums[f"{value_col}_slope"] = _safe_divide(sums["_xy"], sums["_xx"])
    return sums[["Provider", f"{value_col}_slope"]]


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


def _add_dmepos_categorical_derivatives(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return claims
    out = claims.copy()
    if "HCPCS_CD" in out.columns:
        hcpcs = out["HCPCS_CD"].astype(str).str.strip().replace("", "__MISSING__")
        out["HCPCS_prefix1"] = hcpcs.str[:1].replace("", "__MISSING__")
        out["HCPCS_prefix2"] = hcpcs.str[:2].replace("", "__MISSING__")
        out["HCPCS_prefix3"] = hcpcs.str[:3].replace("", "__MISSING__")
        if "Place_Of_Srvc" in out.columns:
            place = out["Place_Of_Srvc"].astype(str).str.strip().replace("", "__MISSING__")
            out["HCPCS_place_combo"] = hcpcs + "__place__" + place
            out["HCPCS_prefix3_place_combo"] = out["HCPCS_prefix3"] + "__place__" + place
        if "HCPCS_Drug_Ind" in out.columns:
            drug = out["HCPCS_Drug_Ind"].astype(str).str.strip().replace("", "__MISSING__")
            out["HCPCS_drug_combo"] = hcpcs + "__drug__" + drug
            out["HCPCS_prefix3_drug_combo"] = out["HCPCS_prefix3"] + "__drug__" + drug
        if "Rfrg_Prvdr_Spclty_Desc" in out.columns:
            specialty = out["Rfrg_Prvdr_Spclty_Desc"].astype(str).str.strip().replace("", "__MISSING__")
            out["specialty_hcpcs_prefix3_combo"] = specialty + "__hcpcs3__" + out["HCPCS_prefix3"]
        if "Rfrg_Prvdr_State_Abrvtn" in out.columns:
            state = out["Rfrg_Prvdr_State_Abrvtn"].astype(str).str.strip().replace("", "__MISSING__")
            out["state_hcpcs_prefix2_combo"] = state + "__hcpcs2__" + out["HCPCS_prefix2"]
    if {"RBCS_Id", "Place_Of_Srvc"}.issubset(out.columns):
        rbcs = out["RBCS_Id"].astype(str).str.strip().replace("", "__MISSING__")
        place = out["Place_Of_Srvc"].astype(str).str.strip().replace("", "__MISSING__")
        out["RBCS_place_combo"] = rbcs + "__place__" + place
    if {"RBCS_Id", "Year"}.issubset(out.columns):
        rbcs = out["RBCS_Id"].astype(str).str.strip().replace("", "__MISSING__")
        year = out["Year"].astype(str).str.strip().replace("", "__MISSING__")
        out["RBCS_year_combo"] = rbcs + "__year__" + year
    return out


def _dmepos_numeric_columns(claims: pd.DataFrame) -> list[str]:
    preferred = [
        "Rfrg_Prvdr_RUCA",
        "Tot_Suplrs",
        "Tot_Suplr_Benes",
        "Tot_Suplr_Clms",
        "Tot_Suplr_Srvcs",
        "Avg_Suplr_Sbmtd_Chrg",
        "Avg_Suplr_Mdcr_Alowd_Amt",
        "Avg_Suplr_Mdcr_Pymt_Amt",
        "Avg_Suplr_Mdcr_Stdzd_Amt",
        "submitted_charge_total",
        "allowed_amount_total",
        "payment_amount_total",
        "standardized_amount_total",
        "charge_allowed_gap",
        "allowed_payment_gap",
        "standardized_payment_gap",
        "service_per_bene",
        "claim_per_bene",
        "payment_per_bene",
        "allowed_per_bene",
        "submitted_per_bene",
        "service_per_claim",
        "Year",
    ]
    return [
        c
        for c in preferred + [c for c in claims.columns if c.startswith("ratio_")]
        if c in claims.columns
    ]


def _dmepos_categorical_columns(claims: pd.DataFrame) -> list[str]:
    preferred = [
        "HCPCS_Desc",
        "HCPCS_CD",
        "Rfrg_Prvdr_Spclty_Cd",
        "Rfrg_Prvdr_Spclty_Desc",
        "Rfrg_Prvdr_State_FIPS",
        "Rfrg_Prvdr_State_Abrvtn",
        "Rfrg_Prvdr_Zip5",
        "RBCS_Id",
        "RBCS_Desc",
        "RBCS_Lvl",
        "Rfrg_Prvdr_Spclty_Srce",
        "Rfrg_Prvdr_Ent_Cd",
        "Rfrg_Prvdr_Crdntls",
        "Rfrg_Prvdr_Mdcr_Prtcptg_Ind",
        "来源文件",
        "source_file",
        "Rfrg_Prvdr_City",
        "Rfrg_Prvdr_RUCA_Desc",
        "Rfrg_Prvdr_RUCA_Cat",
        "Rfrg_Prvdr_RUCA",
        "Suplr_Rentl_Ind",
        "HCPCS_Drug_Ind",
        "Place_Of_Srvc",
        "HCPCS_prefix1",
        "HCPCS_prefix2",
        "HCPCS_prefix3",
        "HCPCS_place_combo",
        "HCPCS_prefix3_place_combo",
        "HCPCS_drug_combo",
        "HCPCS_prefix3_drug_combo",
        "specialty_hcpcs_prefix3_combo",
        "state_hcpcs_prefix2_combo",
        "RBCS_place_combo",
        "RBCS_year_combo",
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


def _dmepos_indicator_levels(train_claims: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    if train_claims.empty:
        return {}
    max_levels = {
        "Rfrg_Prvdr_Spclty_Desc": 80,
        "Rfrg_Prvdr_State_Abrvtn": 60,
        "Rfrg_Prvdr_State_FIPS": 60,
        "Rfrg_Prvdr_Zip5": 80,
        "Rfrg_Prvdr_RUCA": 20,
        "Rfrg_Prvdr_Ent_Cd": 10,
        "Rfrg_Prvdr_Crdntls": 50,
        "Rfrg_Prvdr_Mdcr_Prtcptg_Ind": 10,
        "HCPCS_Drug_Ind": 10,
        "Place_Of_Srvc": 10,
        "Year": 20,
        "HCPCS_CD": 120,
    }
    provider_labels = pd.Series(dtype=int)
    if "ProviderLabel" in train_claims.columns:
        provider_labels = (
            train_claims.assign(Provider=lambda x: x["Provider"].astype(str))
            .groupby("Provider")["ProviderLabel"]
            .max()
        )
        provider_labels = pd.to_numeric(provider_labels, errors="coerce").fillna(0).astype(int)

    levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        limit = max_levels.get(col)
        if not limit or col not in train_claims.columns:
            continue
        values = train_claims[["Provider", col]].copy()
        values["Provider"] = values["Provider"].astype(str)
        values[col] = values[col].astype(str).str.strip().replace("", "__MISSING__")
        values = values.drop_duplicates()
        common = values[col].value_counts().head(limit).index.astype(str).tolist()
        selected = list(common)
        if not provider_labels.empty:
            scored = values.copy()
            scored["ProviderLabel"] = scored["Provider"].map(provider_labels).fillna(0).astype(int)
            stats = scored.groupby(col)["ProviderLabel"].agg(["sum", "mean", "count"])
            target_levels = (
                stats[stats["sum"] >= 1]
                .sort_values(["sum", "mean", "count"], ascending=[False, False, False])
                .head(limit)
                .index.astype(str)
                .tolist()
            )
            selected.extend(target_levels)
        levels[col] = list(dict.fromkeys(selected))[:limit]
    return levels


def _dmepos_hcpcs_profile_documents(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty or not {"Provider", "HCPCS_CD"}.issubset(claims.columns):
        return pd.DataFrame(columns=["Provider", "hcpcs_profile_doc"])

    cols = [
        c
        for c in [
            "Provider",
            "HCPCS_CD",
            "Rfrg_Prvdr_Spclty_Desc",
            "HCPCS_Drug_Ind",
            "Place_Of_Srvc",
            "Year",
            "Rfrg_Prvdr_State_Abrvtn",
        ]
        if c in claims.columns
    ]
    work = claims[cols].copy()
    work["Provider"] = work["Provider"].astype(str)
    code = work["HCPCS_CD"].astype(str).str.strip().replace("", "__MISSING__")
    code_safe = code.map(lambda value: _safe_feature_token(value).lower())
    prefix1 = code.str[:1].map(lambda value: _safe_feature_token(value).lower())
    prefix2 = code.str[:2].map(lambda value: _safe_feature_token(value).lower())
    prefix3 = code.str[:3].map(lambda value: _safe_feature_token(value).lower())
    tokens = (
        "hcpcs_" + code_safe
        + " hcpcs_p1_" + prefix1
        + " hcpcs_p2_" + prefix2
        + " hcpcs_p3_" + prefix3
    )

    if "Place_Of_Srvc" in work.columns:
        place = work["Place_Of_Srvc"].astype(str).str.strip().map(lambda value: _safe_feature_token(value).lower())
        tokens = tokens + " hcpcs_place_" + code_safe + "_" + place + " place_" + place
    if "HCPCS_Drug_Ind" in work.columns:
        drug = work["HCPCS_Drug_Ind"].astype(str).str.strip().map(lambda value: _safe_feature_token(value).lower())
        tokens = tokens + " hcpcs_drug_" + code_safe + "_" + drug + " drug_" + drug
    if "Rfrg_Prvdr_Spclty_Desc" in work.columns:
        specialty = work["Rfrg_Prvdr_Spclty_Desc"].astype(str).str.strip().map(lambda value: _safe_feature_token(value, max_length=32).lower())
        tokens = tokens + " specialty_" + specialty + " specialty_hcpcs_p3_" + specialty + "_" + prefix3
    if "Year" in work.columns:
        year = work["Year"].astype(str).str.strip().map(lambda value: _safe_feature_token(value).lower())
        tokens = tokens + " year_" + year + " year_hcpcs_p2_" + year + "_" + prefix2
    if "Rfrg_Prvdr_State_Abrvtn" in work.columns:
        state = work["Rfrg_Prvdr_State_Abrvtn"].astype(str).str.strip().map(lambda value: _safe_feature_token(value).lower())
        tokens = tokens + " state_hcpcs_p2_" + state + "_" + prefix2

    docs = (
        pd.DataFrame({"Provider": work["Provider"], "hcpcs_profile_doc": tokens})
        .groupby("Provider", sort=True)["hcpcs_profile_doc"]
        .agg(" ".join)
        .reset_index()
    )
    return docs


def _dmepos_hcpcs_profile_context(train_claims: pd.DataFrame) -> dict[str, object]:
    docs = _dmepos_hcpcs_profile_documents(train_claims)
    if docs.empty or "ProviderLabel" not in train_claims.columns:
        return {}
    provider_labels = (
        train_claims.assign(Provider=lambda x: x["Provider"].astype(str))
        .groupby("Provider")["ProviderLabel"]
        .max()
    )
    provider_labels = pd.to_numeric(provider_labels, errors="coerce").fillna(0).astype(int)
    docs = docs.merge(provider_labels.rename("ProviderLabel"), on="Provider", how="left").fillna({"ProviderLabel": 0})
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=1,
        token_pattern=r"[^ ]+",
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(docs["hcpcs_profile_doc"].astype(str))
    return {
        "docs": docs,
        "matrix": matrix,
        "labels": docs["ProviderLabel"].astype(int).to_numpy(),
        "providers": docs["Provider"].astype(str).to_numpy(),
        "vectorizer": vectorizer,
    }


def _hcpcs_profile_similarity_frame(
    matrix: object,
    providers: np.ndarray,
    reference_matrix: object,
    reference_labels: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame({"Provider": providers.astype(str)})
    if len(reference_labels) == 0 or matrix.shape[0] == 0:
        return out

    positive_mask = reference_labels.astype(int) == 1
    negative_mask = reference_labels.astype(int) == 0
    positive_matrix = reference_matrix[positive_mask]
    negative_matrix = reference_matrix[negative_mask]

    if positive_matrix.shape[0] == 0:
        for col in [
            "hcpcs_pos_centroid_sim",
            "hcpcs_neg_centroid_sim",
            "hcpcs_centroid_diff",
            "hcpcs_centroid_ratio",
            "hcpcs_pos_nn_max",
            "hcpcs_pos_nn_mean",
            "hcpcs_pos_nn_top3_mean",
            "hcpcs_pos_nn_share_ge_010",
            "hcpcs_pos_nn_share_ge_020",
            "hcpcs_pos_nn_share_ge_030",
        ]:
            out[col] = 0.0
        return out

    positive_centroid = np.asarray(positive_matrix.mean(axis=0))
    negative_centroid = np.asarray(negative_matrix.mean(axis=0)) if negative_matrix.shape[0] else np.zeros_like(positive_centroid)
    out["hcpcs_pos_centroid_sim"] = cosine_similarity(matrix, positive_centroid).ravel()
    out["hcpcs_neg_centroid_sim"] = cosine_similarity(matrix, negative_centroid).ravel()
    out["hcpcs_centroid_diff"] = out["hcpcs_pos_centroid_sim"] - out["hcpcs_neg_centroid_sim"]
    out["hcpcs_centroid_ratio"] = _safe_divide(out["hcpcs_pos_centroid_sim"], out["hcpcs_neg_centroid_sim"] + 1e-6)

    max_values: list[float] = []
    mean_values: list[float] = []
    top3_values: list[float] = []
    share_010: list[float] = []
    share_020: list[float] = []
    share_030: list[float] = []
    chunk_size = 1000
    for start in range(0, matrix.shape[0], chunk_size):
        sim = cosine_similarity(matrix[start : start + chunk_size], positive_matrix)
        max_values.extend(sim.max(axis=1).astype(float))
        mean_values.extend(sim.mean(axis=1).astype(float))
        top_k = min(3, sim.shape[1])
        top3_values.extend(np.sort(sim, axis=1)[:, -top_k:].mean(axis=1).astype(float))
        share_010.extend((sim >= 0.10).mean(axis=1).astype(float))
        share_020.extend((sim >= 0.20).mean(axis=1).astype(float))
        share_030.extend((sim >= 0.30).mean(axis=1).astype(float))

    out["hcpcs_pos_nn_max"] = max_values
    out["hcpcs_pos_nn_mean"] = mean_values
    out["hcpcs_pos_nn_top3_mean"] = top3_values
    out["hcpcs_pos_nn_share_ge_010"] = share_010
    out["hcpcs_pos_nn_share_ge_020"] = share_020
    out["hcpcs_pos_nn_share_ge_030"] = share_030
    return out.fillna(0)


def _dmepos_hcpcs_profile_features(
    claims: pd.DataFrame,
    context: dict[str, object],
    *,
    split_key: str,
) -> pd.DataFrame:
    if not context:
        return pd.DataFrame(columns=["Provider"])

    train_matrix = context["matrix"]
    train_labels = context["labels"]
    train_providers = context["providers"]
    vectorizer = context["vectorizer"]

    if split_key == "train":
        if len(np.unique(train_labels)) < 2 or np.bincount(train_labels.astype(int)).min() < 2:
            return _hcpcs_profile_similarity_frame(train_matrix, train_providers, train_matrix, train_labels)
        features: list[pd.DataFrame] = []
        n_splits = min(5, int(np.bincount(train_labels.astype(int)).min()))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        for train_idx, valid_idx in splitter.split(np.zeros(len(train_labels)), train_labels):
            features.append(
                _hcpcs_profile_similarity_frame(
                    train_matrix[valid_idx],
                    train_providers[valid_idx],
                    train_matrix[train_idx],
                    train_labels[train_idx],
                )
            )
        return pd.concat(features, ignore_index=True)

    docs = _dmepos_hcpcs_profile_documents(claims)
    if docs.empty:
        return pd.DataFrame(columns=["Provider"])
    matrix = vectorizer.transform(docs["hcpcs_profile_doc"].astype(str))
    return _hcpcs_profile_similarity_frame(matrix, docs["Provider"].astype(str).to_numpy(), train_matrix, train_labels)


def _dmepos_oof_risk_values(train_claims: pd.DataFrame, categorical_cols: list[str], global_rate: float, smoothing: float = 35.0) -> dict[str, pd.Series]:
    if train_claims.empty or "ProviderLabel" not in train_claims.columns:
        return {}
    y = pd.to_numeric(train_claims["ProviderLabel"], errors="coerce").fillna(0).astype(int).to_numpy()
    if len(np.unique(y)) < 2 or np.bincount(y).min() < 2:
        return {}
    n_splits = min(5, int(np.bincount(y).min()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    out: dict[str, pd.Series] = {}
    hcpcs_combo_cols = [
        c
        for c in categorical_cols
        if c.startswith("HCPCS_prefix") or c in {"HCPCS_place_combo", "HCPCS_drug_combo"}
    ]
    risk_cols = [
        c
        for c in dict.fromkeys(
            [
            "HCPCS_CD",
            *hcpcs_combo_cols,
            "RBCS_Id",
            "RBCS_Lvl",
            "Rfrg_Prvdr_Spclty_Cd",
            "Rfrg_Prvdr_Spclty_Desc",
            "Rfrg_Prvdr_State_Abrvtn",
            "Rfrg_Prvdr_State_FIPS",
            "Rfrg_Prvdr_Zip5",
            "Rfrg_Prvdr_Ent_Cd",
            "Rfrg_Prvdr_Crdntls",
            "Rfrg_Prvdr_Mdcr_Prtcptg_Ind",
            "Suplr_Rentl_Ind",
            "HCPCS_Drug_Ind",
            "Place_Of_Srvc",
            "Year",
            ]
        )
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
    categorical_indicator_levels: dict[str, list[str]] | None = None,
    *,
    split_key: str,
) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame()
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
        derived_numeric: dict[str, pd.Series] = {}
        for col in numeric_cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"
            min_col = f"{col}_min"
            max_col = f"{col}_max"
            if mean_col in result.columns and std_col in result.columns:
                derived_numeric[f"{col}_cv"] = _safe_divide(result[std_col], result[mean_col].abs())
            if min_col in result.columns and max_col in result.columns:
                max_values = pd.to_numeric(result[max_col], errors="coerce").fillna(0.0)
                min_values = pd.to_numeric(result[min_col], errors="coerce").fillna(0.0)
                derived_numeric[f"{col}_range"] = max_values - min_values
        if derived_numeric:
            result = pd.concat([result, pd.DataFrame(derived_numeric)], axis=1)

    risk_values_by_col = _dmepos_oof_risk_values(work, categorical_cols, global_risk) if split_key == "train" else {}
    provider_sizes = grouped.size()
    for col in categorical_cols:
        result = result.merge(
            _provider_categorical_stats(
                work,
                col,
                provider_sizes,
                categorical_risk_maps,
                global_risk,
                (categorical_frequency_maps or {}).get(col, {}),
                risk_values_by_col.get(col) if split_key == "train" else None,
                (categorical_indicator_levels or {}).get(col, []),
            ),
            on="Provider",
            how="left",
        )

    if "Year" in work.columns and numeric_cols:
        trend_cols = [
            c
            for c in [
                "Tot_Suplrs",
                "Tot_Suplr_Benes",
                "Tot_Suplr_Clms",
                "Tot_Suplr_Srvcs",
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
        if not yearly.empty:
            trend = yearly.groupby("Provider", sort=False)["Year"].agg(["min", "max"]).reset_index()
            trend["time_span"] = trend["max"] - trend["min"]
            trend = trend[["Provider", "time_span"]]
            recent_year = yearly.groupby("Provider", sort=False)["Year"].transform("max")
            recent = yearly[yearly["Year"] == recent_year].groupby("Provider", sort=False)[trend_cols].mean().reset_index()
            recent = recent.rename(columns={col: f"{col}_recent" for col in trend_cols})
            trend = trend.merge(recent, on="Provider", how="left")
            for col in trend_cols:
                trend = trend.merge(_provider_year_slope(yearly, col), on="Provider", how="left")
            result = result.merge(trend, on="Provider", how="left")

    if not label_df.empty and {"Provider", "ProviderLabel"}.issubset(label_df.columns):
        labels = label_df[["Provider", "ProviderLabel"]].copy()
        labels["Provider"] = labels["Provider"].astype(str)
        result = result.merge(labels, on="Provider", how="left")
    return result.fillna(0)


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
                if PREPROCESSING.get("enable_hcpcs_combo_features", False):
                    claims = _add_dmepos_categorical_derivatives(claims)
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
        risk_maps, global_risk_rate = _dmepos_risk_maps(
            train_claims_for_risk,
            dmepos_categorical_cols,
            smoothing=PREPROCESSING["risk_smoothing"],
        )
        frequency_maps = _dmepos_frequency_maps(train_claims_for_risk, dmepos_categorical_cols)
        indicator_levels = (
            _dmepos_indicator_levels(train_claims_for_risk, dmepos_categorical_cols)
            if PREPROCESSING.get("enable_hcpcs_combo_features", False)
            else {}
        )
        hcpcs_profile_context = _dmepos_hcpcs_profile_context(train_claims_for_risk)
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
        indicator_levels = {}
        hcpcs_profile_context = {}

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
                    indicator_levels,
                    split_key=split_key,
                )
                hcpcs_profile_features = _dmepos_hcpcs_profile_features(
                    claims,
                    hcpcs_profile_context,
                    split_key=split_key,
                )
                if not hcpcs_profile_features.empty:
                    static_features = static_features.merge(hcpcs_profile_features, on="Provider", how="left").fillna(0)
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
