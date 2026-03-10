"""
models.py
LightGBM, CatBoost, XGBoost, MLP — CV-обучение и предсказание.

Публичный API (именно эти имена ожидает main.py):
  train_lgb_cv          → (fold_results, models)
  train_lgb_full        → model
  train_catboost_cv     → (fold_results, models)
  train_xgb_cv          → (fold_results, models)
  train_mlp_cv          → (fold_results, models)
  predict_lgb_ensemble  → np.ndarray
  predict_cb_ensemble   → np.ndarray
  predict_xgb_ensemble  → np.ndarray

fold_results — список dict на каждый фолд:
  {
    "fold":       int,
    "val_idx":    np.ndarray,
    "oof_preds":  np.ndarray,
    "oof_labels": np.ndarray,
    "pr_auc":     float,
    "model":      обученная модель,
  }
"""

import numpy as np
import logging
import warnings
import json
from pathlib import Path
from typing import Tuple, Optional, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import (
    CV_FOLDS, TIMESTAMP_COL, LABEL_COL, EVENT_ID_COL, CLIENT_ID_COL,
    LGB_PARAMS, LGB_N_ROUNDS, LGB_EARLY_STOP, CB_PARAMS,
    MODELS_DIR, RED, YELLOW, SEED,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if y_true.sum() == 0:
        return 0.0
    return average_precision_score(y_true, y_score)


def _temporal_splits(df, feature_cols: list) -> list:
    """
    Делит по CV_FOLDS из конфига.
    Возвращает список (X_train, y_train, X_val, y_val, val_mask).
    """
    import polars as pl

    splits = []
    for fold_cfg in CV_FOLDS:
        train_mask = pl.col(TIMESTAMP_COL) <= pl.lit(fold_cfg["train_end"]).str.to_datetime("%Y-%m-%d")
        val_mask   = (
            (pl.col(TIMESTAMP_COL) >= pl.lit(fold_cfg["val_start"]).str.to_datetime("%Y-%m-%d")) &
            (pl.col(TIMESTAMP_COL) <= pl.lit(fold_cfg["val_end"]).str.to_datetime("%Y-%m-%d"))
        )

        # Только размеченные строки
        labeled_mask = pl.col(LABEL_COL).is_not_null()
        red_green_mask = pl.col(LABEL_COL).is_in([0, 1])  # без жёлтых в таргете

        train_df = df.filter(train_mask & labeled_mask & red_green_mask)
        val_df   = df.filter(val_mask   & labeled_mask & red_green_mask)

        if train_df.height == 0 or val_df.height == 0:
            logger.warning(f"Empty fold: {fold_cfg}")
            continue

        X_tr = train_df.select(feature_cols).fill_nan(0).fill_null(0).to_numpy().astype(np.float32)
        y_tr = (train_df[LABEL_COL] == RED).cast(int).to_numpy().astype(np.float32)
        X_val = val_df.select(feature_cols).fill_nan(0).fill_null(0).to_numpy().astype(np.float32)
        y_val = (val_df[LABEL_COL] == RED).cast(int).to_numpy().astype(np.float32)

        splits.append((X_tr, y_tr, X_val, y_val, val_df))

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lgb_cv(
    df,                         # polars DataFrame с фичами
    feature_cols: list,
    params: dict = None,
    use_yellow_as_negative: bool = False,
) -> Tuple[list, list]:
    """
    Обучает LightGBM по временны́м фолдам из CV_FOLDS.
    Возвращает (fold_results, models).
    """
    import lightgbm as lgb

    lgb_params = (params or LGB_PARAMS).copy()
    lgb_params.setdefault("seed", SEED)
    early_stop = lgb_params.pop("early_stopping_rounds", LGB_EARLY_STOP)
    n_rounds   = lgb_params.pop("n_estimators", LGB_N_ROUNDS)

    splits       = _temporal_splits(df, feature_cols)
    fold_results = []
    models       = []

    for fold_idx, (X_tr, y_tr, X_val, y_val, val_df) in enumerate(splits):
        logger.info(f"\n  Fold {fold_idx+1}/{len(splits)} | "
                    f"train={len(X_tr):,} | val={len(X_val):,} | "
                    f"fraud_rate_train={y_tr.mean():.4f}")

        dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feature_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        callbacks = [
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(200),
        ]

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        oof_preds = model.predict(X_val, num_iteration=model.best_iteration)
        score     = _pr_auc(y_val, oof_preds)

        logger.info(f"  Fold {fold_idx+1} PR-AUC: {score:.5f} | "
                    f"best_iter: {model.best_iteration}")

        # Сохраняем модель
        model_path = MODELS_DIR / f"lgb_fold{fold_idx}.txt"
        model.save_model(str(model_path))

        fold_results.append({
            "fold":       fold_idx,
            "val_idx":    np.arange(len(y_val)),   # относительные индексы
            "oof_preds":  oof_preds,
            "oof_labels": y_val,
            "pr_auc":     score,
            "model":      model,
        })
        models.append(model)

    scores = [r["pr_auc"] for r in fold_results]
    logger.info(f"\n  LGB CV PR-AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

    return fold_results, models


def train_lgb_full(
    df,
    feature_cols: list,
    params: dict = None,
    n_rounds: int = None,
) -> object:
    """
    Переобучает LightGBM на ВСЁМ train-датасете (для финального сабмита).
    n_rounds берётся из среднего best_iteration по фолдам или указывается явно.
    """
    import lightgbm as lgb
    import polars as pl

    lgb_params = (params or LGB_PARAMS).copy()
    lgb_params.pop("early_stopping_rounds", None)
    lgb_params.pop("n_estimators", None)

    labeled = df.filter(
        pl.col(LABEL_COL).is_not_null() & pl.col(LABEL_COL).is_in([0, 1])
    )
    X = labeled.select(feature_cols).fill_nan(0).fill_null(0).to_numpy().astype(np.float32)
    y = (labeled[LABEL_COL] == RED).cast(int).to_numpy().astype(np.float32)

    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model  = lgb.train(lgb_params, dtrain, num_boost_round=n_rounds or LGB_N_ROUNDS)

    model.save_model(str(MODELS_DIR / "lgb_full.txt"))
    logger.info(f"Full LGB trained on {len(X):,} rows")
    return model


def predict_lgb_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Усредняет предсказания всех LGB-моделей из фолдов."""
    preds = np.column_stack([
        m.predict(X, num_iteration=m.best_iteration if hasattr(m, "best_iteration") else 0)
        for m in models
    ])
    return preds.mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# CATBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_catboost_cv(
    df,
    feature_cols: list,
    params: dict = None,
) -> Tuple[list, list]:
    """Обучает CatBoost по временны́м фолдам."""
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        logger.error("catboost not installed. Run: pip install catboost")
        return [], []

    import polars as pl
    cb_params = (params or CB_PARAMS).copy()
    cb_params.setdefault("random_seed", SEED)
    # GPU может не быть, ставим CPU как fallback
    if cb_params.get("task_type") == "GPU":
        try:
            import subprocess
            subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        except Exception:
            cb_params["task_type"] = "CPU"
            logger.warning("No GPU found, CatBoost will use CPU")

    splits       = _temporal_splits(df, feature_cols)
    fold_results = []
    models       = []

    for fold_idx, (X_tr, y_tr, X_val, y_val, val_df) in enumerate(splits):
        logger.info(f"\n  CB Fold {fold_idx+1}/{len(splits)} | "
                    f"train={len(X_tr):,} | val={len(X_val):,}")

        model = CatBoostClassifier(**cb_params)
        model.fit(
            Pool(X_tr,  y_tr),
            eval_set=Pool(X_val, y_val),
            use_best_model=True,
            verbose=cb_params.get("verbose", 200),
        )

        oof_preds = model.predict_proba(X_val)[:, 1]
        score     = _pr_auc(y_val, oof_preds)
        logger.info(f"  CB Fold {fold_idx+1} PR-AUC: {score:.5f}")

        model.save_model(str(MODELS_DIR / f"cb_fold{fold_idx}.cbm"))

        fold_results.append({
            "fold":       fold_idx,
            "val_idx":    np.arange(len(y_val)),
            "oof_preds":  oof_preds,
            "oof_labels": y_val,
            "pr_auc":     score,
            "model":      model,
        })
        models.append(model)

    scores = [r["pr_auc"] for r in fold_results]
    logger.info(f"\n  CB CV PR-AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return fold_results, models


def predict_cb_ensemble(models: list, X: np.ndarray) -> np.ndarray:
    """Усредняет предсказания всех CB-моделей."""
    preds = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    return preds.mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb_cv(
    df,
    feature_cols: list,
    params: dict = None,
) -> Tuple[list, list]:
    """Обучает XGBoost по временны́м фолдам."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        return [], []

    from configs.config import SCALE_POS_WEIGHT
    xgb_params = params or {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "max_depth":        8,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "tree_method":      "hist",
        "seed":             SEED,
        "verbosity":        0,
    }
    n_rounds   = xgb_params.pop("n_estimators", LGB_N_ROUNDS)
    early_stop = xgb_params.pop("early_stopping_rounds", LGB_EARLY_STOP)

    splits       = _temporal_splits(df, feature_cols)
    fold_results = []
    models       = []

    for fold_idx, (X_tr, y_tr, X_val, y_val, val_df) in enumerate(splits):
        logger.info(f"\n  XGB Fold {fold_idx+1}/{len(splits)} | "
                    f"train={len(X_tr):,} | val={len(X_val):,}")

        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feature_cols)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stop,
            verbose_eval=200,
        )

        oof_preds = model.predict(dval)
        score     = _pr_auc(y_val, oof_preds)
        logger.info(f"  XGB Fold {fold_idx+1} PR-AUC: {score:.5f}")

        model.save_model(str(MODELS_DIR / f"xgb_fold{fold_idx}.json"))

        fold_results.append({
            "fold":       fold_idx,
            "val_idx":    np.arange(len(y_val)),
            "oof_preds":  oof_preds,
            "oof_labels": y_val,
            "pr_auc":     score,
            "model":      model,
        })
        models.append(model)

    scores = [r["pr_auc"] for r in fold_results]
    logger.info(f"\n  XGB CV PR-AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return fold_results, models


def predict_xgb_ensemble(models: list, X: np.ndarray, feature_cols: list) -> np.ndarray:
    """Усредняет предсказания всех XGB-моделей."""
    import xgboost as xgb
    dmat  = xgb.DMatrix(X, feature_names=feature_cols)
    preds = np.column_stack([m.predict(dmat) for m in models])
    return preds.mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# MLP (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════

def train_mlp_cv(
    df,
    feature_cols: list,
    hidden_dims: list = None,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 4096,
    patience: int = 10,
    dropout: float = 0.3,
) -> Tuple[list, list]:
    """Обучает MLP по временны́м фолдам."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("PyTorch not available, skipping MLP")
        return [], []

    hidden_dims  = hidden_dims or [512, 256, 128, 64]
    splits       = _temporal_splits(df, feature_cols)
    fold_results = []
    models_list  = []
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"MLP device: {device}")

    for fold_idx, (X_tr, y_tr, X_val, y_val, val_df) in enumerate(splits):
        logger.info(f"\n  MLP Fold {fold_idx+1}/{len(splits)}")

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s  = scaler.transform(X_val).astype(np.float32)

        # Строим MLP
        dims   = [X_tr.shape[1]] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers += [nn.BatchNorm1d(dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        model = nn.Sequential(*layers).to(device)

        pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())]).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr)),
            batch_size=batch_size, shuffle=True,
        )

        best_score, best_state, pat = 0.0, None, 0
        for epoch in range(epochs):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(Xb).squeeze(-1), yb).backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                logits = model(torch.FloatTensor(X_val_s).to(device)).squeeze(-1).cpu().numpy()
            score = _pr_auc(y_val, logits)

            if score > best_score:
                best_score = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= patience:
                    logger.info(f"    Early stop at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_preds = model(torch.FloatTensor(X_val_s).to(device)).squeeze(-1).cpu().numpy()

        logger.info(f"  MLP Fold {fold_idx+1} PR-AUC: {best_score:.5f}")

        fold_results.append({
            "fold":       fold_idx,
            "val_idx":    np.arange(len(y_val)),
            "oof_preds":  oof_preds,
            "oof_labels": y_val,
            "pr_auc":     best_score,
            "model":      (model, scaler),
        })
        models_list.append((model, scaler))

    scores = [r["pr_auc"] for r in fold_results]
    if scores:
        logger.info(f"\n  MLP CV PR-AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    return fold_results, models_list


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(models: list, feature_cols: list, model_type: str) -> dict:
    """Усредняет важность признаков по всем фолдам."""
    importances = np.zeros(len(feature_cols))
    for m in models:
        if model_type == "lgb":
            imp = m.feature_importance(importance_type="gain")
        elif model_type == "catboost":
            imp = m.get_feature_importance()
        elif model_type == "xgb":
            scores = m.get_fscore()
            imp = np.array([scores.get(f, 0) for f in feature_cols])
        else:
            continue
        importances += imp
    importances /= max(len(models), 1)
    return dict(sorted(zip(feature_cols, importances), key=lambda x: -x[1]))