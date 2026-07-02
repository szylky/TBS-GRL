from __future__ import annotations

import numpy as np


                                                                           
GLOBAL_HYPERPARAMETERS = {
    "random": {
        "seed": {"value": 42, "source": "original preprocessing/temporal/group seed"},
    },
    "preprocessing": {
        "dmepos_target_smoothing": {"value": 35.0, "source": "original DMEPOS target smoothing"},
        "generic_target_smoothing": {"value": 20.0, "source": "original generic target smoothing"},
        "graph_overlap_threshold": {"value": 2, "source": "original graph overlap threshold"},
        "rare_category_frequency_threshold": {"value": 0.001, "source": "original rare category threshold"},
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
        "relation_top_k": {"value": 17, "source": "fixed from installed @5 optimum trial_0010"},
        "relation_min_sim": {"value": 0.5759018403940501, "source": "fixed from installed @5 optimum trial_0010"},
        "business_weight_mode": {"value": "binary", "source": "fixed from installed @5 optimum trial_0010"},
        "relation_max_degree": {"value": 32, "source": "fixed from installed @5 optimum trial_0010"},
        "relation_weight_transform": {"value": "raw", "source": "fixed from installed @5 optimum trial_0010"},
    },
    "group_classification": {
        "loky_max_cpu_count": {"value": "8", "source": "original group_classification_module default"},
        "ensemble_seeds": {"value": (7, 17, 29), "source": "fixed from installed @5 optimum trial_0010"},
        "oof_folds": {"value": 3, "source": "fixed from installed @5 optimum trial_0010"},
        "threshold_min": {"value": 0.008617447206791902, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "threshold_max": {"value": 0.6864135108169754, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "threshold_steps": {"value": 60, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "fixed_threshold": {"value": 0.0086, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "adaptive_rate_multiplier": {"value": 4.855243373581211, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "negative_sample_ratio": {"value": 45, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "balanced_metric_name": {"value": "balanced_four", "source": "custom threshold metric name"},
        "balanced_metric_components": {"value": ("precision", "f1", "recall", "gmean"), "source": "custom threshold metric components"},
        "balanced_spread_penalty": {"value": 0.2142982243978887, "source": "fixed from installed @5/PR-AUC best trial_0023"},
        "hist_gbdt_params": {
            "value": {
                "max_iter": 1100,
                "learning_rate": 0.03603992793651861,
                "max_leaf_nodes": 23,
                "l2_regularization": 0.06673803226381043,
                "class_weight": None,
            },
            "source": "fixed from installed @5/PR-AUC best trial_0023",
        },
        "classifier_backend": {"value": "hist_gbdt", "source": "fixed backend for installed @5 optimum trial_0010"},
        "group_classifier_n_jobs": {"value": 4, "source": "original GROUP_CLASSIFIER_N_JOBS default"},
        "xgb_max_depth": {"value": 0, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_tree_method": {"value": "hist", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_device": {"value": "cpu", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "lgbm_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "lgbm_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "catboost_depth": {"value": 6, "source": "unused unless GROUP_CLASSIFIER_BACKEND=catboost"},
        "use_graph_features": {"value": True, "source": "fixed from installed @5/PR-AUC best trial_0023"},
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
