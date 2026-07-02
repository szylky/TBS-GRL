from __future__ import annotations

import numpy as np


                                                                          
                                                                           
GLOBAL_HYPERPARAMETERS = {
    "random": {
        "seed": {"value": 42, "source": "original preprocessing/temporal/group seed"},
    },
    "preprocessing": {
        "graph_overlap_threshold": {"value": 2, "source": "original graph overlap threshold"},
        "risk_smoothing": {"value": 20.0, "source": "original Part_B risk smoothing"},
        "rare_category_frequency_threshold": {"value": 0.001, "source": "original rare category threshold"},
        "enable_hcpcs_combo_features": {"value": False, "source": "guarded feature experiment; disabled because v1 hurt top-k ranking"},
    },
    "temporal_modeling": {
        "model_type": {"value": "gru", "source": "original temporal encoder default"},
        "hidden_dim": {"value": 64, "source": "original TemporalModelWrapper hidden_dim"},
        "dropout": {"value": 0.2, "source": "original temporal dropout"},
        "transformer_heads": {"value": 4, "source": "original transformer head count"},
        "transformer_layers": {"value": 2, "source": "original transformer layer count"},
        "transformer_max_len": {"value": 512, "source": "original transformer max_len"},
        "learning_rate": {"value": 1e-3, "source": "original torch.optim.Adam lr"},
        "epochs": {"value": 100, "source": "original temporal epoch count"},
        "relation_top_k": {"value": 12, "source": "original build_temporal_outputs default"},
        "relation_min_sim": {"value": 0.55, "source": "original build_temporal_outputs default"},
        "business_weight_mode": {"value": "raw", "source": "original build_temporal_outputs default"},
        "relation_max_degree": {"value": None, "source": "original build_temporal_outputs default"},
        "relation_weight_transform": {"value": "raw", "source": "original build_temporal_outputs default"},
        "similarity_chunk_size": {"value": 512, "source": "memory-safe implementation detail"},
        "dense_graph_provider_limit": {"value": 5000, "source": "prevents accidental giant dense graph output"},
    },
    "group_classification": {
        "loky_max_cpu_count": {"value": "8", "source": "original group_classification_module default"},
        "ensemble_seeds": {"value": (7, 17, 29), "source": "original DEFAULT_SEEDS"},
        "oof_folds": {"value": 5, "source": "original OOF_FOLDS"},
        "threshold_min": {"value": 0.01, "source": "original THRESHOLD_GRID lower bound"},
        "threshold_max": {"value": 0.9, "source": "original THRESHOLD_GRID upper bound"},
        "threshold_steps": {"value": 90, "source": "original THRESHOLD_GRID steps"},
        "train_threshold_override": {"value": None, "source": "preserves original Train.py default"},
        "default_test_threshold": {"value": 0.0007986277941364481, "source": "current best installed DEFAULT_BALANCED_THRESHOLD"},
        "adaptive_rate_multiplier": {"value": 56.0, "source": "current best installed ADAPTIVE_RATE_MULTIPLIER"},
        "negative_sample_ratio": {"value": 20, "source": "Part_B raw_signal_temporal transfer 20260615: best @5 candidate"},
        "balanced_metric_name": {"value": "f1", "source": "original run_group_classification default"},
        "feature_policy": {"value": "raw_signal_temporal", "source": "Part_D-style non-leaky raw-signal + temporal feature policy"},
        "raw_feature_keywords": {
            "value": (
                "ratio_",
                "_share",
                "_cv",
                "_range",
                "_nunique",
                "row_",
                "hhi",
                "entropy",
                "amount",
                "charge",
                "payment",
                "claim",
                "service",
                "bene",
            ),
            "source": "Part_B transferred from Part_D rank/raw signal feature filtering",
        },
        "hist_gbdt_params": {
            "value": {
                "max_iter": 360,
                "learning_rate": 0.022,
                "max_leaf_nodes": 11,
                "l2_regularization": 0.16,
                "class_weight": "balanced",
            },
            "source": "Part_B raw_signal_temporal transfer 20260615 n20_i360_l11",
        },
        "classifier_backend": {"value": "hist_gbdt", "source": "original classifier"},
        "group_classifier_n_jobs": {"value": 4, "source": "advanced classifier optional backend default"},
        "xgb_max_depth": {"value": 0, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_tree_method": {"value": "hist", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_device": {"value": "cpu", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "lgbm_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "lgbm_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "catboost_depth": {"value": 6, "source": "unused unless GROUP_CLASSIFIER_BACKEND=catboost"},
        "use_graph_features": {"value": False, "source": "preserves original Part_B feature set"},
    },
    "screening": {
        "review_rates": {"value": (0.005, 0.01, 0.02, 0.03, 0.05, 0.10), "source": "original screening review rates"},
        "top_ks": {"value": (10, 20, 50, 100, 200), "source": "original screening top-k reports"},
    },
}


def hp(section: str, name: str):
    return GLOBAL_HYPERPARAMETERS[section][name]["value"]


RANDOM_SEED = hp("random", "seed")

PREPROCESSING = {name: item["value"] for name, item in GLOBAL_HYPERPARAMETERS["preprocessing"].items()}
TEMPORAL = {name: item["value"] for name, item in GLOBAL_HYPERPARAMETERS["temporal_modeling"].items()}
GROUP = {name: item["value"] for name, item in GLOBAL_HYPERPARAMETERS["group_classification"].items()}
SCREENING = {name: item["value"] for name, item in GLOBAL_HYPERPARAMETERS["screening"].items()}

THRESHOLD_GRID = np.round(
    np.linspace(GROUP["threshold_min"], GROUP["threshold_max"], GROUP["threshold_steps"]),
    2,
)
