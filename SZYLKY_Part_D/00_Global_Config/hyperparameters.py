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
        "relation_top_k": {"value": 12, "source": "original relation_top_k default"},
        "relation_min_sim": {"value": 0.55, "source": "original relation_min_sim default"},
        "business_weight_mode": {"value": "raw", "source": "original business_weight_mode default"},
        "relation_max_degree": {"value": None, "source": "original relation_max_degree default"},
        "relation_weight_transform": {"value": "raw", "source": "original relation_weight_transform default"},
    },
    "group_classification": {
        "loky_max_cpu_count": {"value": "8", "source": "original group_classification_module default"},
        "ensemble_seeds": {"value": (5, 13, 19, 71, 101), "source": "Part_D SOTA seed ensemble"},
        "oof_folds": {"value": 2, "source": "Part_D tuned OOF folds with fixed threshold"},
        "threshold_min": {"value": 0.01, "source": "original threshold grid min"},
        "threshold_max": {"value": 0.9, "source": "original threshold grid max"},
        "threshold_steps": {"value": 90, "source": "original threshold grid steps"},
        "train_threshold_override": {"value": 0.5, "source": "Part_D tuned threshold for balanced test precision/F1"},
        "adaptive_rate_multiplier": {"value": 0.92, "source": "original adaptive threshold multiplier"},
        "negative_sample_ratio": {"value": 20, "source": "original negative sample ratio"},
        "balanced_metric_name": {"value": "f1", "source": "original Train.py threshold metric default"},
        "balanced_metric_components": {"value": ("precision", "f1", "recall", "gmean"), "source": "original balanced4 components"},
        "feature_policy": {"value": "rank_temporal", "source": "Part_D tuned feature policy"},
        "raw_feature_keywords": {
            "value": (
                "ratio_",
                "partd_",
                "log1p_",
                "_hhi",
                "_share",
                "_recent",
                "_cv",
                "_range",
                "_nunique",
                "row_",
            ),
            "source": "Part_D non-leaky raw feature whitelist",
        },
        "hist_gbdt_params": {
            "value": {
                "max_iter": 550,
                "learning_rate": 0.014,
                "max_leaf_nodes": 31,
                "l2_regularization": 0.08,
                "class_weight": "balanced",
            },
            "source": "Part_D tuned HistGradientBoostingClassifier parameters",
        },
        "classifier_backend": {"value": "hist_gbdt", "source": "original classifier backend"},
        "group_classifier_n_jobs": {"value": 4, "source": "advanced backend compatibility default"},
        "xgb_max_depth": {"value": 0, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_tree_method": {"value": "hist", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_device": {"value": "cpu", "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "xgb_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=xgboost"},
        "lgbm_subsample": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "lgbm_colsample_bytree": {"value": 0.9, "source": "unused unless GROUP_CLASSIFIER_BACKEND=lightgbm"},
        "catboost_depth": {"value": 6, "source": "unused unless GROUP_CLASSIFIER_BACKEND=catboost"},
        "use_graph_features": {"value": False, "source": "disabled to preserve original Part_D feature set"},
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
