"""
models.py — Обучение моделей с временно́й кросс-валидацией
===========================================================
Реализованы:
  - LightGBM   (основная лошадка)
  - CatBoost   (хорошо с категориями)
  - XGBoost    (диверсификация)
  - MLP / TabNet (нейросеть)
  - GRU Sequence Model (история как последовательность)
"""

import polars as pl
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *
from src.validation import (
    FoldResult, iterate_folds, prepare_Xy, prepare_Xy_no_yellow,
    compute_pr_auc, print_cv_summary
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM
# ═══════════════════════════════════════════════════════════════════════════════

def train_lgb_cv(
    df: pl.DataFrame,
    feature_cols: List[str],
    params: Optional[Dict] = None,
    use_yellow_as_negative: bool = False,
    save_models: bool = True,
) -> Tuple[List[FoldResult], List]:
    """
    Обучает LightGBM с временно́й CV.

    use_yellow_as_negative: True = жёлтые в обучение как класс 0
                            False = жёлтые исключены из обучения
    """
    import lightgbm as lgb

    params = params or LGB_PARAMS
    fold_results = []
    models = []

    for fold_idx, train_df, val_df in iterate_folds(df):
        logger.info(f"  Training LightGBM fold {fold_idx+1}...")

        if use_yellow_as_negative:
            X_tr, y_tr = prepare_Xy(train_df, feature_cols)
        else:
            X_tr, y_tr = prepare_Xy_no_yellow(train_df, feature_cols)

        # Для валидации: только Red vs Green (без Yellow)
        val_clean = val_df.filter(pl.col(LABEL_COL) != YELLOW)
        X_val, y_val = prepare_Xy(val_clean, feature_cols)

        pos_weight = max(1, (y_tr == 0).sum() / (y_tr == 1).sum() + 1)
        fold_params = {**params, "scale_pos_weight": pos_weight}

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
            lgb.log_evaluation(200),
        ]

        model = lgb.train(
            fold_params,
            dtrain,
            num_boost_round=LGB_N_ROUNDS,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        pr_auc    = compute_pr_auc(y_val, val_preds)
        logger.info(f"  Fold {fold_idx+1} PR-AUC: {pr_auc:.4f}")

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            val_start=CV_FOLDS[fold_idx]["val_start"],
            val_end=CV_FOLDS[fold_idx]["val_end"],
            pr_auc=pr_auc,
            n_train=len(y_tr),
            n_val=len(y_val),
            n_pos_val=int(y_val.sum()),
            oof_preds=val_preds,
            oof_labels=y_val,
            feature_importance=fi,
        ))
        models.append(model)

        if save_models:
            model.save_model(str(MODELS_DIR / f"lgb_fold{fold_idx+1}.txt"))

    print_cv_summary(fold_results)

    # Сохраняем feature importance
    if fold_results and fold_results[0].feature_importance:
        avg_fi = {}
        for col in feature_cols:
            avg_fi[col] = np.mean([
                r.feature_importance.get(col, 0) for r in fold_results
            ])
        fi_sorted = sorted(avg_fi.items(), key=lambda x: -x[1])
        logger.info("\nTop-20 features by LightGBM gain:")
        for name, score in fi_sorted[:20]:
            logger.info(f"  {name:<40} {score:>10.1f}")

        with open(FEATS_DIR / "lgb_feature_importance.json", "w") as f:
            json.dump(dict(fi_sorted), f, indent=2)

    return fold_results, models


def train_lgb_full(
    df: pl.DataFrame,
    feature_cols: List[str],
    n_rounds: Optional[int] = None,
    params: Optional[Dict] = None,
) -> object:
    """
    Обучает финальную LightGBM модель на ВСЁМ train.
    n_rounds: если None — берём среднее best_iteration из CV.
    """
    import lightgbm as lgb

    params = params or LGB_PARAMS
    X, y = prepare_Xy_no_yellow(df, feature_cols)

    pos_weight = max(1, (y == 0).sum() / (y == 1).sum() + 1)
    full_params = {**params, "scale_pos_weight": pos_weight}

    if n_rounds is None:
        n_rounds = LGB_N_ROUNDS

    logger.info(f"Training LightGBM FULL: {X.shape}, n_rounds={n_rounds}")
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model  = lgb.train(full_params, dtrain, num_boost_round=n_rounds)
    model.save_model(str(MODELS_DIR / "lgb_full.txt"))
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CATBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_catboost_cv(
    df: pl.DataFrame,
    feature_cols: List[str],
    cat_feature_indices: Optional[List[int]] = None,
    params: Optional[Dict] = None,
) -> Tuple[List[FoldResult], List]:
    """
    Обучает CatBoost с временно́й CV.
    CatBoost умеет работать с категориальными признаками напрямую.
    """
    from catboost import CatBoostClassifier, Pool

    params = params or CB_PARAMS
    fold_results = []
    models = []

    for fold_idx, train_df, val_df in iterate_folds(df):
        logger.info(f"  Training CatBoost fold {fold_idx+1}...")

        X_tr, y_tr   = prepare_Xy_no_yellow(train_df, feature_cols)
        val_clean     = val_df.filter(pl.col(LABEL_COL) != YELLOW)
        X_val, y_val  = prepare_Xy(val_clean, feature_cols)

        train_pool = Pool(X_tr,  y_tr,  feature_names=feature_cols,
                         cat_features=cat_feature_indices)
        val_pool   = Pool(X_val, y_val, feature_names=feature_cols,
                         cat_features=cat_feature_indices)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
        )

        val_preds = model.predict_proba(X_val)[:, 1]
        pr_auc    = compute_pr_auc(y_val, val_preds)
        logger.info(f"  Fold {fold_idx+1} PR-AUC: {pr_auc:.4f}")

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            val_start=CV_FOLDS[fold_idx]["val_start"],
            val_end=CV_FOLDS[fold_idx]["val_end"],
            pr_auc=pr_auc,
            n_train=len(y_tr),
            n_val=len(y_val),
            n_pos_val=int(y_val.sum()),
            oof_preds=val_preds,
            oof_labels=y_val,
        ))
        models.append(model)

        if save_models := True:
            model.save_model(str(MODELS_DIR / f"cb_fold{fold_idx+1}.cbm"))

    print_cv_summary(fold_results)
    return fold_results, models


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb_cv(
    df: pl.DataFrame,
    feature_cols: List[str],
    params: Optional[Dict] = None,
) -> Tuple[List[FoldResult], List]:
    """Обучает XGBoost с временно́й CV."""
    import xgboost as xgb

    default_params = {
        "objective":        "binary:logistic",
        "eval_metric":      "aucpr",
        "max_depth":        8,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 50,
        "scale_pos_weight": 30,
        "tree_method":      "hist",
        "seed":             SEED,
        "verbosity":        0,
    }
    params = params or default_params
    fold_results = []
    models = []

    for fold_idx, train_df, val_df in iterate_folds(df):
        logger.info(f"  Training XGBoost fold {fold_idx+1}...")

        X_tr, y_tr  = prepare_Xy_no_yellow(train_df, feature_cols)
        val_clean   = val_df.filter(pl.col(LABEL_COL) != YELLOW)
        X_val, y_val = prepare_Xy(val_clean, feature_cols)

        dtrain = xgb.DMatrix(X_tr,  label=y_tr,  feature_names=feature_cols)
        dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        model = xgb.train(
            params, dtrain,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=200,
        )

        val_preds = model.predict(dval)
        pr_auc    = compute_pr_auc(y_val, val_preds)
        logger.info(f"  Fold {fold_idx+1} PR-AUC: {pr_auc:.4f}")

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            val_start=CV_FOLDS[fold_idx]["val_start"],
            val_end=CV_FOLDS[fold_idx]["val_end"],
            pr_auc=pr_auc, n_train=len(y_tr),
            n_val=len(y_val), n_pos_val=int(y_val.sum()),
            oof_preds=val_preds, oof_labels=y_val,
        ))
        models.append(model)
        model.save_model(str(MODELS_DIR / f"xgb_fold{fold_idx+1}.json"))

    print_cv_summary(fold_results)
    return fold_results, models


# ═══════════════════════════════════════════════════════════════════════════════
# НЕЙРОСЕТЬ (MLP / TabNet)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mlp(input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
    """Строит MLP с BatchNorm, Dropout и Focal-Loss совместимым выходом."""
    import torch
    import torch.nn as nn

    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers += [
            nn.Linear(prev_dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.3),
        ]
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


class FocalLoss:
    """
    Focal Loss для сильного дисбаланса классов.
    alpha ≈ 0.25, gamma ≈ 2 стандартные значения.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t  = prob * targets + (1 - prob) * (1 - targets)
        fl   = self.alpha * (1 - p_t) ** self.gamma * bce
        return fl.mean()


def train_mlp_cv(
    df: pl.DataFrame,
    feature_cols: List[str],
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
) -> Tuple[List[FoldResult], List]:
    """Обучает MLP с временно́й CV, Focal Loss."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not installed, skipping MLP")
        return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"MLP training on: {device}")

    fold_results = []
    models = []

    for fold_idx, train_df, val_df in iterate_folds(df):
        X_tr, y_tr  = prepare_Xy_no_yellow(train_df, feature_cols)
        val_clean   = val_df.filter(pl.col(LABEL_COL) != YELLOW)
        X_val, y_val = prepare_Xy(val_clean, feature_cols)

        # Нормализация
        mean = X_tr.mean(0); std = X_tr.std(0) + 1e-9
        X_tr_n  = (X_tr  - mean) / std
        X_val_n = (X_val - mean) / std

        Xtr_t = torch.FloatTensor(X_tr_n).to(device)
        ytr_t = torch.FloatTensor(y_tr).to(device)
        Xv_t  = torch.FloatTensor(X_val_n).to(device)

        model = build_mlp(len(feature_cols)).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=epochs,
            steps_per_epoch=max(1, len(X_tr) // batch_size)
        )
        criterion = FocalLoss(alpha=0.25, gamma=2.0)

        ds     = TensorDataset(Xtr_t, ytr_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best_pr_auc = 0
        best_state  = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for Xb, yb in loader:
                opt.zero_grad()
                logits = model(Xb).squeeze(1)
                loss   = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total_loss += loss.item()

            # Валидация
            model.eval()
            with torch.no_grad():
                logits_val = model(Xv_t).squeeze(1)
                preds_val  = torch.sigmoid(logits_val).cpu().numpy()

            pr_auc = compute_pr_auc(y_val, preds_val)
            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 5 == 0:
                logger.info(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f} PR-AUC={pr_auc:.4f}")

        # Финальные предсказания
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            final_preds = torch.sigmoid(model(Xv_t).squeeze(1)).cpu().numpy()

        pr_auc = compute_pr_auc(y_val, final_preds)
        logger.info(f"  MLP Fold {fold_idx+1} best PR-AUC: {pr_auc:.4f}")

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            val_start=CV_FOLDS[fold_idx]["val_start"],
            val_end=CV_FOLDS[fold_idx]["val_end"],
            pr_auc=pr_auc, n_train=len(y_tr),
            n_val=len(y_val), n_pos_val=int(y_val.sum()),
            oof_preds=final_preds, oof_labels=y_val,
        ))
        models.append((model, mean, std))

        torch.save(model.state_dict(), MODELS_DIR / f"mlp_fold{fold_idx+1}.pt")

    print_cv_summary(fold_results)
    return fold_results, models


# ═══════════════════════════════════════════════════════════════════════════════
# GRU SEQUENCE MODEL (история транзакций клиента)
# ═══════════════════════════════════════════════════════════════════════════════

class TransactionGRU:
    """
    GRU, принимающий историю последних N транзакций клиента.
    Каждая транзакция — вектор числовых и cat-embedding фичей.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        seq_len: int = 50,
        dropout: float = 0.3,
    ):
        try:
            import torch
            import torch.nn as nn

            class GRUModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = nn.GRU(
                        input_dim, hidden_dim,
                        num_layers=n_layers,
                        batch_first=True,
                        dropout=dropout if n_layers > 1 else 0,
                        bidirectional=False,   # Нельзя bidirectional — look-ahead!
                    )
                    self.head = nn.Sequential(
                        nn.Linear(hidden_dim, 64),
                        nn.GELU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                    )

                def forward(self, x):
                    out, _ = self.gru(x)
                    last   = out[:, -1, :]   # последний шаг = текущая операция
                    return self.head(last).squeeze(-1)

            self.model = GRUModel()
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim
        except ImportError:
            self.model = None
            logger.warning("PyTorch not available for GRU model")

    def build_sequences(
        self,
        df: pl.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Для каждой операции строим последовательность из
        предыдущих seq_len операций ТОГО ЖЕ клиента.
        """
        df_sorted = df.sort([CLIENT_ID_COL, TIMESTAMP_COL])
        sequences = []
        labels    = []
        event_ids = []

        for client_id, client_df in df_sorted.group_by(CLIENT_ID_COL):
            ops = client_df.select(feature_cols).fill_null(0).fill_nan(0).to_numpy()
            lbs = (client_df[LABEL_COL] == RED).cast(pl.Int8).to_numpy()
            eids = client_df[EVENT_ID_COL].to_numpy()

            n = len(ops)
            for i in range(n):
                start = max(0, i - self.seq_len + 1)
                seq   = ops[start:i+1]
                # Padding слева нулями
                pad_len = self.seq_len - len(seq)
                if pad_len > 0:
                    seq = np.vstack([np.zeros((pad_len, seq.shape[1])), seq])
                sequences.append(seq)
                labels.append(lbs[i])
                event_ids.append(eids[i])

        return np.array(sequences), np.array(labels), np.array(event_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# ПРЕДСКАЗАНИЕ НА ТЕСТЕ (усреднение по фолдам)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_lgb_ensemble(
    models: List,
    X_test: np.ndarray,
) -> np.ndarray:
    """Усредняет предсказания всех LGB моделей (из разных фолдов)."""
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration)
    return preds / len(models)


def predict_cb_ensemble(models: List, X_test: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict_proba(X_test)[:, 1]
    return preds / len(models)


def predict_xgb_ensemble(models: List, X_test: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    import xgboost as xgb
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    preds = np.zeros(len(X_test))
    for model in models:
        preds += model.predict(dtest)
    return preds / len(models)
