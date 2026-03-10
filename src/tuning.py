"""
tuning.py — Optuna HPO для LightGBM и CatBoost.

Публичный API (ожидает main.py):
  tune_lightgbm(df, feature_cols, n_trials) → best_params dict
  tune_catboost(df, feature_cols, n_trials) → best_params dict
"""

import numpy as np
import logging
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    CV_FOLDS, LABEL_COL, RED, LGB_PARAMS, CB_PARAMS,
    OUTPUT_DIR, SEED,
)
from src.validation import compute_pr_auc

logger = logging.getLogger(__name__)


def _get_splits(df, feature_cols):
    """Переиспользуем сплиттер из models.py."""
    from src.models import _temporal_splits
    return _temporal_splits(df, feature_cols)


# ──────────────────────────────────────────────────────────────────
# LightGBM tuning
# ──────────────────────────────────────────────────────────────────

def tune_lightgbm(
    df,
    feature_cols: list,
    n_trials: int = 100,
    save_path: str = None,
) -> dict:
    """
    Optuna HPO для LightGBM. Сохраняет лучшие параметры в JSON.
    """
    try:
        import optuna
        import lightgbm as lgb
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna or lightgbm not installed")
        return {}

    splits = _get_splits(df, feature_cols)

    def objective(trial):
        params = {
            "objective":        "binary",
            "metric":           "average_precision",
            "verbosity":        -1,
            "boosting_type":    "gbdt",
            "seed":             SEED,
            "num_leaves":       trial.suggest_int("num_leaves", 31, 511),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "lambda_l1":        trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2":        trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
        }

        fold_scores = []
        for X_tr, y_tr, X_val, y_val, _ in splits:
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
            model  = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            fold_scores.append(compute_pr_auc(y_val, preds))

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["_best_pr_auc"] = study.best_value
    logger.info(f"Best LGB PR-AUC: {study.best_value:.5f}")
    logger.info(f"Best LGB params: {best}")

    save_path = save_path or str(OUTPUT_DIR / "best_lgb_params.json")
    with open(save_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Saved to {save_path}")

    return best


# ──────────────────────────────────────────────────────────────────
# CatBoost tuning
# ──────────────────────────────────────────────────────────────────

def tune_catboost(
    df,
    feature_cols: list,
    n_trials: int = 50,
    save_path: str = None,
) -> dict:
    """
    Optuna HPO для CatBoost. Сохраняет лучшие параметры в JSON.
    """
    try:
        import optuna
        from catboost import CatBoostClassifier, Pool
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.error("optuna or catboost not installed")
        return {}

    splits = _get_splits(df, feature_cols)

    def objective(trial):
        params = {
            "loss_function":    "Logloss",
            "eval_metric":      "AUC",
            "random_seed":      SEED,
            "verbose":          0,
            "task_type":        "CPU",
            "iterations":       500,
            "od_type":          "Iter",
            "od_wait":          50,
            "depth":            trial.suggest_int("depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength":  trial.suggest_float("random_strength", 0.1, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 5, 50),
        }

        fold_scores = []
        for X_tr, y_tr, X_val, y_val, _ in splits:
            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr),
                eval_set=Pool(X_val, y_val),
                use_best_model=True,
                verbose=False,
            )
            preds = model.predict_proba(X_val)[:, 1]
            fold_scores.append(compute_pr_auc(y_val, preds))

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["_best_pr_auc"] = study.best_value
    logger.info(f"Best CB PR-AUC: {study.best_value:.5f}")

    save_path = save_path or str(OUTPUT_DIR / "best_cb_params.json")
    with open(save_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info(f"Saved to {save_path}")

    return best