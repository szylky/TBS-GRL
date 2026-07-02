from __future__ import annotations

"""Provider-level temporal sequence modeling.

This module builds per-Provider time series features, encodes them with either
an GRU or a lightweight Transformer encoder, exports temporal embeddings,
predicts a future risk score, and constructs a Provider association graph from
both temporal similarity and business overlap.

Outputs (per split):
- provider_temporal_features.csv
- provider_temporal_embedding.csv
- provider_future_risk.csv
- provider_temporal_relation_graph.csv
- provider_temporal_relation_edges.csv
- predicted_Cg_next.csv
- temporal_training_summary.csv
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from hyperparameters import RANDOM_SEED, TEMPORAL
from logging_utils import log_line, log_progress, log_stage, print_table


def _set_random_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


         
                                      
                                                               
                            
                                                                    
                                                   
 
                                         
                                
                                  
                                                                  
                             
class GRUTemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
                                                                
                                                           
                                                                                                   
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)                              
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h_seq, _ = self.gru(x)                                                         
        h_last = h_seq[:, -1, :]                               
        h_last = self.dropout(h_last)                                 
        risk = torch.sigmoid(self.risk_head(h_last).squeeze(-1))
        return h_last, risk

                 
                                              
                                                                                 
                            
                                                            
                                                                             
                                                                                                                               
                                                                   
                                                   
 
                                         
                                
                                                  
                             
                              
                                                                  
                             
class TransformerTemporalEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)        
                  
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

                                                
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(n_heads, hidden_dim // 8)),                         
            batch_first=True,
            dropout=dropout,                     
            activation='relu'                  
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)                              
        self.risk_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)                              
        h = h + self.pos_emb[:, :h.shape[1], :]          
        h = self.encoder(h)          
        h_last = h[:, -1, :]            
        h_last = self.dropout(h_last)       
        risk = torch.sigmoid(self.risk_head(h_last).squeeze(-1))
        return h_last, risk


class TemporalModelWrapper(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = TEMPORAL["hidden_dim"],
        model_type: str = TEMPORAL["model_type"],
        dropout: float = TEMPORAL["dropout"],
    ):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == "transformer":
            self.encoder = TransformerTemporalEncoder(
                input_dim,
                hidden_dim,
                n_heads=TEMPORAL["transformer_heads"],
                num_layers=TEMPORAL["transformer_layers"],
                dropout=dropout,
                max_len=TEMPORAL["transformer_max_len"],
            )
        else:
            self.encoder = GRUTemporalEncoder(input_dim, hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor):
        return self.encoder(x)


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path).fillna(0)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _find_temporal_base(split_dir: Path) -> pd.DataFrame:
    for name in ["provider_temporal_features.csv", "claims_enriched.csv"]:
        df = _load_csv(split_dir / name)
        if not df.empty:
            return df
    return pd.DataFrame()


def _build_provider_time_features(split_dir: Path) -> pd.DataFrame:
    cached = _load_csv(split_dir / "provider_temporal_features.csv")
    if not cached.empty:
        return cached

    claims = _load_csv(split_dir / "claims_enriched.csv")
    if claims.empty or not {"Provider", "TimeWindow"}.issubset(claims.columns):
        return pd.DataFrame()

    numeric_cols = [
        c
        for c in claims.columns
        if c not in {"BeneID", "ClaimID", "Provider", "ClaimType", "TimeWindow", "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "PotentialFraud", "ProviderLabel"}
        and pd.api.types.is_numeric_dtype(claims[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()

    agg = claims.groupby(["Provider", "TimeWindow"], as_index=False)[numeric_cols].mean()
    agg.to_csv(split_dir / "provider_temporal_features.csv", index=False)
    return agg


def _pivot_time_series(df: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    if df.empty:
        return [], [], np.zeros((0, 0, 0), dtype=np.float32)

    provider_col = "Provider"
    time_col = "TimeWindow"
    feature_cols = [c for c in df.columns if c not in {provider_col, time_col}]
    times = sorted(df[time_col].astype(str).unique().tolist())
    providers = sorted(df[provider_col].astype(str).unique().tolist())
    arr = np.zeros((len(providers), len(times), len(feature_cols)), dtype=np.float32)
    prov_idx = {p: i for i, p in enumerate(providers)}
    time_idx = {t: i for i, t in enumerate(times)}

    feature_frame = df[feature_cols].copy()
    for col in feature_frame.columns:
        if pd.api.types.is_numeric_dtype(feature_frame[col]):
            continue
        normalized = feature_frame[col].astype(str).str.strip().str.lower()
        mapped = normalized.map({"yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0})
        feature_frame[col] = pd.to_numeric(mapped.where(mapped.notna(), feature_frame[col]), errors="coerce")

    feature_frame = feature_frame.fillna(0.0)

    for idx, row in df.iterrows():
        p = str(row[provider_col])
        t = str(row[time_col])
        if p not in prov_idx or t not in time_idx:
            continue
        arr[prov_idx[p], time_idx[t], :] = feature_frame.loc[idx].to_numpy(dtype=np.float32)
    return providers, times, arr


def _normalize_ts(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    out = arr.copy().astype(np.float32)
    mean = out.mean(axis=(0, 1), keepdims=True)
    std = out.std(axis=(0, 1), keepdims=True)
    out = (out - mean) / np.where(std > 1e-6, std, 1.0)
    return np.nan_to_num(out)


def _mean_pool_time(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return arr.mean(axis=1)


def _pairwise_cosine(x: np.ndarray, top_k: int = 12, min_sim: float = 0.55) -> pd.DataFrame:
    if x.size == 0:
        return pd.DataFrame(columns=["Provider_src", "Provider_dst", "similarity"])
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    xn = x / norms
    sim = xn @ xn.T
    np.fill_diagonal(sim, 0.0)
    rows = []
    for i in range(sim.shape[0]):
        idx = np.argsort(-sim[i])[:top_k]
        for j in idx:
            if sim[i, j] >= min_sim:
                rows.append((i, j, float(sim[i, j])))
    return pd.DataFrame(rows, columns=["Provider_src", "Provider_dst", "similarity"])


def _business_edges(split_dir: Path) -> pd.DataFrame:
    claims = _load_csv(split_dir / "claims_enriched.csv")
    if claims.empty or not {"Provider", "BeneID"}.issubset(claims.columns):
        return pd.DataFrame(columns=["Provider_src", "Provider_dst", "shared_bene_count"])
    gb = claims.groupby(["BeneID", "Provider"], as_index=False).size()
    rows = []
    for bene_id, sub in gb.groupby("BeneID", sort=False):
        providers = sub["Provider"].astype(str).tolist()
        for i in range(len(providers)):
            for j in range(i + 1, len(providers)):
                rows.append((providers[i], providers[j], 1))
                rows.append((providers[j], providers[i], 1))
    if not rows:
        return pd.DataFrame(columns=["Provider_src", "Provider_dst", "shared_bene_count"])
    out = pd.DataFrame(rows, columns=["Provider_src", "Provider_dst", "shared_bene_count"])
    return out.groupby(["Provider_src", "Provider_dst"], as_index=False)["shared_bene_count"].sum()


def _transform_edge_weights(weights: pd.Series, mode: str) -> pd.Series:
    values = pd.to_numeric(weights, errors="coerce").fillna(1.0).clip(lower=0.0)
    mode = mode.lower()
    if mode == "log1p":
        return np.log1p(values)
    if mode == "clipped_p95":
        cap = float(values.quantile(0.95)) if len(values) else 1.0
        return values.clip(upper=max(cap, 1e-6))
    if mode == "binary":
        return (values > 0).astype(float)
    return values


def _limit_max_degree(edges: pd.DataFrame, max_degree: int | None) -> pd.DataFrame:
    if max_degree is None or max_degree <= 0 or edges.empty:
        return edges
    return (
        edges.sort_values(["Provider_src", "weight"], ascending=[True, False])
        .groupby("Provider_src", as_index=False, sort=False)
        .head(max_degree)
        .reset_index(drop=True)
    )


def _predict_next_edges(
    providers: List[str],
    embeddings: np.ndarray,
    business: pd.DataFrame,
    top_k: int = 12,
    min_sim: float = 0.55,
    business_weight_mode: str = "raw",
    max_degree: int | None = None,
    weight_transform: str = "raw",
) -> pd.DataFrame:
    sim_edges = _pairwise_cosine(embeddings, top_k=top_k, min_sim=min_sim)
    if sim_edges.empty and business.empty:
        return pd.DataFrame(columns=["Provider_src", "Provider_dst", "weight"])

    prov_idx = {p: i for i, p in enumerate(providers)}
    edge_rows = []
    for _, r in sim_edges.iterrows():
        i = int(r.iloc[0])
        j = int(r.iloc[1])
        s = float(r.iloc[2])
        edge_rows.append((providers[i], providers[j], s))
    if business_weight_mode.lower() != "off" and not business.empty:
        for _, r in business.iterrows():
            src = str(r["Provider_src"])
            dst = str(r["Provider_dst"])
            if src not in prov_idx or dst not in prov_idx:
                continue
            weight = 1.0 if business_weight_mode.lower() == "binary" else float(r["shared_bene_count"])
            edge_rows.append((src, dst, weight))
    if not edge_rows:
        return pd.DataFrame(columns=["Provider_src", "Provider_dst", "weight"])

    edges = pd.DataFrame(edge_rows, columns=["Provider_src", "Provider_dst", "weight"])
    edges["weight"] = _transform_edge_weights(edges["weight"], weight_transform)
    edges = edges.groupby(["Provider_src", "Provider_dst"], as_index=False)["weight"].max()
    edges = _limit_max_degree(edges, max_degree)
    reverse_edges = edges.rename(columns={"Provider_src": "Provider_dst", "Provider_dst": "Provider_src"})
    sym_edges = pd.concat([edges, reverse_edges], ignore_index=True)
    sym_edges = sym_edges.groupby(["Provider_src", "Provider_dst"], as_index=False)["weight"].sum()
    sym_edges["weight"] = sym_edges["weight"] * 0.5
    return sym_edges[sym_edges["weight"] > 0].reset_index(drop=True)


def _edges_to_adjacency(providers: List[str], edges: pd.DataFrame) -> pd.DataFrame:
    mat = np.zeros((len(providers), len(providers)), dtype=np.float32)
    if edges.empty:
        return pd.DataFrame(mat, index=providers, columns=providers)
    prov_idx = {p: i for i, p in enumerate(providers)}
    for _, r in edges.iterrows():
        src = prov_idx[str(r["Provider_src"])]
        dst = prov_idx[str(r["Provider_dst"])]
        mat[src, dst] = max(mat[src, dst], float(r["weight"]))
    np.fill_diagonal(mat, 0.0)
    return pd.DataFrame(mat, index=providers, columns=providers)


def build_temporal_outputs(
    data_root: str | Path,
    split: str,
    output_root: str | Path,
    model_output_root: str | Path | None = None,
    load_from_train_model: bool = False,
    write_dense_graph: bool = False,
    relation_top_k: int = TEMPORAL["relation_top_k"],
    relation_min_sim: float = TEMPORAL["relation_min_sim"],
    business_weight_mode: str = TEMPORAL["business_weight_mode"],
    relation_max_degree: int | None = TEMPORAL["relation_max_degree"],
    relation_weight_transform: str = TEMPORAL["relation_weight_transform"],
) -> Dict[str, object]:
    _set_random_seed()
    split_dir = Path(data_root) / split
    out_dir = Path(output_root) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(model_output_root) / "temporal" / "train" if model_output_root else out_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "temporal_provider_model.pt"

    temporal_df = _build_provider_time_features(split_dir)
    if temporal_df.empty:
        pd.DataFrame([{"split": split, "snapshot_count": 0, "provider_count": 0}]).to_csv(out_dir / "temporal_training_summary.csv", index=False)
        return {"provider_temporal_embedding": pd.DataFrame(), "provider_future_risk": pd.DataFrame(), "relation_graph": pd.DataFrame()}

    providers, times, arr = _pivot_time_series(temporal_df)
    arr = _normalize_ts(arr)
    n_providers, t_steps, f_dim = arr.shape
    x = torch.from_numpy(arr)
    print_table(
        f"时序输入摘要 {split}",
        [
            {
                "providers": n_providers,
                "time_steps": t_steps,
                "feature_dim": f_dim,
                "mode": "load_model" if load_from_train_model else "train",
            }
        ],
    )

    model = TemporalModelWrapper(
        input_dim=f_dim,
        hidden_dim=TEMPORAL["hidden_dim"],
        model_type=TEMPORAL["model_type"],
        dropout=TEMPORAL["dropout"],
    )

    if load_from_train_model:
        with log_stage(f"加载时序模型 {split}"):
            if not model_path.exists():
                raise FileNotFoundError(f"未找到训练好的时序模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        with log_stage(f"训练时序模型 {split}"):
            target = _mean_pool_time(arr).sum(axis=1)
            target = (target - target.mean()) / (target.std() + 1e-6)
            y_risk = 1.0 / (1.0 + np.exp(-target))
            optimizer = torch.optim.Adam(model.parameters(), lr=TEMPORAL["learning_rate"])
            target_tensor = torch.from_numpy(y_risk.astype(np.float32))
            model.train()
            total_epochs = TEMPORAL["epochs"]
            for epoch in range(total_epochs):
                optimizer.zero_grad()
                emb, risk = model(x)
                loss = torch.mean((risk - target_tensor) ** 2)
                loss.backward()
                optimizer.step()
                if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == total_epochs:
                    log_progress("时序训练", epoch + 1, total_epochs, extra=f"loss={loss.item():.6f}")
            torch.save(model.state_dict(), model_path)

    model.eval()
    with torch.no_grad():
        emb, risk = model(x)
    emb_np = emb.numpy().astype(np.float32)
    risk_np = risk.numpy().astype(np.float32)

    hidden_cols = [f"temporal_hidden_{i:03d}" for i in range(emb_np.shape[1])]
    emb_df = pd.DataFrame(emb_np, columns=hidden_cols)
    emb_df.insert(0, "Provider", providers)
    emb_df.to_csv(out_dir / "provider_temporal_embedding.csv", index=False)
    emb_df.to_csv(out_dir / "provider_temporal_hidden.csv", index=False)

    risk_df = pd.DataFrame({"Provider": providers, "future_risk_score": risk_np, "future_strength_score": _mean_pool_time(arr).sum(axis=1)})
    risk_df.to_csv(out_dir / "provider_future_risk.csv", index=False)

    with log_stage(f"构建时序关系图 {split}"):
        business = _business_edges(split_dir)
        rel_edges = _predict_next_edges(
            providers,
            emb_np,
            business,
            top_k=relation_top_k,
            min_sim=relation_min_sim,
            business_weight_mode=business_weight_mode,
            max_degree=relation_max_degree,
            weight_transform=relation_weight_transform,
        )
        rel_edges.to_csv(out_dir / "provider_temporal_relation_edges.csv", index=False)

        if write_dense_graph:
            relation_graph = _edges_to_adjacency(providers, rel_edges)
            relation_graph.to_csv(out_dir / "provider_temporal_relation_graph.csv")
            relation_graph.to_csv(out_dir / "predicted_Cg_next.csv")
        else:
            relation_graph = pd.DataFrame()
            log_line("已跳过 dense graph CSV 写出，仅保留 edge list", tag="SKIP")

    summary = pd.DataFrame([
        {
            "split": split,
            "provider_count": int(n_providers),
            "time_steps": int(t_steps),
            "feature_dim_per_window": int(f_dim),
            "embedding_dim": int(emb_np.shape[1]),
            "relation_edge_count": int(len(rel_edges)),
            "relation_top_k": int(relation_top_k),
            "relation_min_sim": float(relation_min_sim),
            "business_weight_mode": business_weight_mode,
            "relation_max_degree": relation_max_degree,
            "relation_weight_transform": relation_weight_transform,
            "risk_score_mean": float(risk_np.mean()) if len(risk_np) else 0.0,
            "risk_score_std": float(risk_np.std()) if len(risk_np) else 0.0,
        }
    ])
    summary.to_csv(out_dir / "temporal_training_summary.csv", index=False)

    print_table(
        f"时序输出摘要 {split}",
        [
            {
                "providers": n_providers,
                "embedding_dim": emb_np.shape[1],
                "relation_edges": len(rel_edges),
                "risk_mean": float(risk_np.mean()) if len(risk_np) else 0.0,
                "risk_std": float(risk_np.std()) if len(risk_np) else 0.0,
            }
        ],
    )
    return {
        "provider_temporal_embedding": emb_df,
        "provider_future_risk": risk_df,
        "relation_graph": relation_graph,
        "relation_edges": rel_edges,
    }
