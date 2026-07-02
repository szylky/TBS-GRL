from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from hyperparameters import GROUP


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _balanced_scale_pos_weight(y: pd.Series | np.ndarray | None) -> float:
    if y is None:
        return 1.0
    values = np.asarray(y).astype(int)
    pos = int(np.sum(values == 1))
    neg = int(np.sum(values == 0))
    if pos <= 0:
        return 1.0
    return max(1.0, neg / pos)


def _hist_gbdt(random_state: int, params: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(params["max_iter"]),
        learning_rate=float(params["learning_rate"]),
        max_leaf_nodes=int(params["max_leaf_nodes"]),
        l2_regularization=float(params["l2_regularization"]),
        class_weight=params.get("class_weight"),
        random_state=random_state,
    )


def build_classifier_pipeline(random_state: int, y: pd.Series | np.ndarray | None, hist_params: dict[str, Any]) -> Pipeline:
    backend = os.environ.get("GROUP_CLASSIFIER_BACKEND", GROUP["classifier_backend"]).strip().lower()
    backend = {"hgbdt": "hist_gbdt", "hist": "hist_gbdt", "xgb": "xgboost", "lgbm": "lightgbm"}.get(backend, backend)

    estimator: Any
    if backend == "xgboost":
        try:
            from xgboost import XGBClassifier

            estimator = XGBClassifier(
                n_estimators=int(os.environ.get("XGB_N_ESTIMATORS", hist_params["max_iter"])),
                learning_rate=float(os.environ.get("XGB_LEARNING_RATE", hist_params["learning_rate"])),
                max_leaves=int(os.environ.get("XGB_MAX_LEAVES", hist_params["max_leaf_nodes"])),
                max_depth=int(os.environ.get("XGB_MAX_DEPTH", str(GROUP["xgb_max_depth"]))),
                grow_policy="lossguide",
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method=os.environ.get("XGB_TREE_METHOD", GROUP["xgb_tree_method"]),
                device=os.environ.get("XGB_DEVICE", GROUP["xgb_device"]),
                subsample=float(os.environ.get("XGB_SUBSAMPLE", str(GROUP["xgb_subsample"]))),
                colsample_bytree=float(os.environ.get("XGB_COLSAMPLE_BYTREE", str(GROUP["xgb_colsample_bytree"]))),
                reg_lambda=float(os.environ.get("XGB_REG_LAMBDA", hist_params["l2_regularization"])),
                scale_pos_weight=_balanced_scale_pos_weight(y),
                n_jobs=int(os.environ.get("GROUP_CLASSIFIER_N_JOBS", str(GROUP["group_classifier_n_jobs"]))),
                random_state=random_state,
            )
        except ImportError:
            print("[WARN] xgboost is not installed; falling back to hist_gbdt.")
            backend = "hist_gbdt"
            estimator = _hist_gbdt(random_state, hist_params)
    elif backend == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                n_estimators=int(os.environ.get("LGBM_N_ESTIMATORS", hist_params["max_iter"])),
                learning_rate=float(os.environ.get("LGBM_LEARNING_RATE", hist_params["learning_rate"])),
                num_leaves=int(os.environ.get("LGBM_NUM_LEAVES", hist_params["max_leaf_nodes"])),
                objective="binary",
                class_weight=hist_params.get("class_weight"),
                subsample=float(os.environ.get("LGBM_SUBSAMPLE", str(GROUP["lgbm_subsample"]))),
                colsample_bytree=float(os.environ.get("LGBM_COLSAMPLE_BYTREE", str(GROUP["lgbm_colsample_bytree"]))),
                reg_lambda=float(os.environ.get("LGBM_REG_LAMBDA", hist_params["l2_regularization"])),
                n_jobs=int(os.environ.get("GROUP_CLASSIFIER_N_JOBS", str(GROUP["group_classifier_n_jobs"]))),
                random_state=random_state,
            )
        except ImportError:
            print("[WARN] lightgbm is not installed; falling back to hist_gbdt.")
            backend = "hist_gbdt"
            estimator = _hist_gbdt(random_state, hist_params)
    elif backend == "catboost":
        try:
            from catboost import CatBoostClassifier

            estimator = CatBoostClassifier(
                iterations=int(os.environ.get("CATBOOST_ITERATIONS", hist_params["max_iter"])),
                learning_rate=float(os.environ.get("CATBOOST_LEARNING_RATE", hist_params["learning_rate"])),
                depth=int(os.environ.get("CATBOOST_DEPTH", str(GROUP["catboost_depth"]))),
                l2_leaf_reg=float(os.environ.get("CATBOOST_L2_LEAF_REG", hist_params["l2_regularization"])),
                loss_function="Logloss",
                eval_metric="PRAUC",
                auto_class_weights="Balanced" if hist_params.get("class_weight") == "balanced" else None,
                random_seed=random_state,
                verbose=False,
                thread_count=int(os.environ.get("GROUP_CLASSIFIER_N_JOBS", str(GROUP["group_classifier_n_jobs"]))),
            )
        except ImportError:
            print("[WARN] catboost is not installed; falling back to hist_gbdt.")
            backend = "hist_gbdt"
            estimator = _hist_gbdt(random_state, hist_params)
    else:
        backend = "hist_gbdt"
        estimator = _hist_gbdt(random_state, hist_params)

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", estimator)])
    pipe.classifier_backend_ = backend
    return pipe


def classifier_name(models: list[Pipeline]) -> str:
    if not models:
        return os.environ.get("GROUP_CLASSIFIER_BACKEND", GROUP["classifier_backend"])
    return str(getattr(models[0], "classifier_backend_", "hist_gbdt"))


def merge_graphsage_features(feats: pd.DataFrame, base_root: str | Path, split: str) -> pd.DataFrame:
    if not _env_flag("GROUP_USE_GRAPH_FEATURES", GROUP["use_graph_features"]):
        return feats

    base_root = Path(base_root)
    graph_path = base_root / split / "provider_temporal_relation_edges.csv"
    emb_path = base_root / split / "provider_temporal_embedding.csv"
    risk_path = base_root / split / "provider_future_risk.csv"
    if not graph_path.exists() or not emb_path.exists():
        return feats

    edges = pd.read_csv(graph_path, low_memory=False)
    if edges.empty or not {"Provider_src", "Provider_dst", "weight"}.issubset(edges.columns):
        return feats

    node = pd.read_csv(emb_path, low_memory=False)
    if risk_path.exists():
        risk = pd.read_csv(risk_path, low_memory=False)
        node = node.merge(risk, on="Provider", how="left")
    if node.empty or "Provider" not in node.columns:
        return feats

    node["Provider"] = node["Provider"].astype(str)
    node_cols = [c for c in node.columns if c != "Provider"]
    node[node_cols] = node[node_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

    edges = edges[["Provider_src", "Provider_dst", "weight"]].copy()
    edges["Provider_src"] = edges["Provider_src"].astype(str)
    edges["Provider_dst"] = edges["Provider_dst"].astype(str)
    edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0).astype(np.float32)
    edges = edges[edges["weight"] > 0]
    if edges.empty:
        return feats

    merged = feats.copy()
    if "Provider" in merged.columns:
        merged["Provider"] = merged["Provider"].astype(str)

    for prefix, group_col, neighbor_col in (
        ("graphsage_in", "Provider_dst", "Provider_src"),
        ("graphsage_out", "Provider_src", "Provider_dst"),
    ):
        tmp = edges.merge(node, left_on=neighbor_col, right_on="Provider", how="left")
        if tmp.empty:
            continue
        tmp[node_cols] = tmp[node_cols].fillna(0)
        weighted = tmp[node_cols].multiply(tmp["weight"], axis=0)
        summed = weighted.groupby(tmp[group_col]).sum()
        denom = tmp.groupby(group_col)["weight"].sum().replace(0, np.nan)
        agg = summed.div(denom, axis=0).fillna(0)
        agg.columns = [f"{prefix}_{c}" for c in agg.columns]
        agg.insert(0, "Provider", agg.index.astype(str))
        agg[f"{prefix}_degree"] = tmp.groupby(group_col).size().reindex(agg["Provider"]).fillna(0).to_numpy()
        agg[f"{prefix}_weight_sum"] = denom.reindex(agg["Provider"]).fillna(0).to_numpy()
        merged = merged.merge(agg.reset_index(drop=True), on="Provider", how="left")

    graph_cols = [c for c in merged.columns if c.startswith("graphsage_")]
    if graph_cols:
        merged[graph_cols] = merged[graph_cols].fillna(0).astype(np.float32)
    return merged
