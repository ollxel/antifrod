"""
validation.py — Временна́я кросс-валидация без утечки данных
=============================================================
ПРАВИЛО: при обучении модели на фолде X —
фичи валидационных строк считаются ТОЛЬКО по данным до val_start.
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional
from dataclasses import dataclass
from sklearn.metrics import average_precision_score

import sys
sys.path.append(str(Path(__file__).parent.parent))
from configs.config import *

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold_idx:    int
    val_start:   str
    val_end:     str
    pr_auc:      float
    n_train:     int
    n_val:       int
    n_pos_val:   int
    oof_preds:   np.ndarray
    oof_labels:  np.ndarray
    feature_importance: Optional[Dict[str, float]] = None


# ═══════════════════════════════════════════════════════════════════════════════
# РАЗБИВКА НА ФОЛДЫ
# ═══════════════════════════════════════════════════════════════════════════════

def temporal_train_val_split(
    df: pl.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    gap_days: int = 0,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Разбивает датасет на train/val по времени.

    gap_days: зазор между train_end и val_start во избежание
              утечки через медленно обновляемые фичи.
    """
    train_cutoff = pl.lit(train_end).str.to_datetime("%Y-%m-%d")
    val_start_dt = pl.lit(val_start).str.to_datetime("%Y-%m-%d")
    val_end_dt   = pl.lit(val_end).str.to_datetime("%Y-%m-%d") + pl.duration(days=1)

    train = df.filter(pl.col(TIMESTAMP_COL) <= train_cutoff)
    val   = df.filter(
        (pl.col(TIMESTAMP_COL) >= val_start_dt)
        & (pl.col(TIMESTAMP_COL) < val_end_dt)
    )

    # Только Red/Green для метрики (без Yellow)
    val_for_metric = val.filter(pl.col(LABEL_COL) != YELLOW)

    logger.info(
        f"  Split: train={train.height:,} | val={val.height:,} "
        f"(red={val_for_metric.filter(pl.col(LABEL_COL)==RED).height:,})"
    )
    return train, val


def iterate_folds(
    df: pl.DataFrame,
    folds: List[Dict] = CV_FOLDS,
) -> Generator[Tuple[int, pl.DataFrame, pl.DataFrame], None, None]:
    """Генератор фолдов для кросс-валидации."""
    for i, fold in enumerate(folds):
        logger.info(f"\nFold {i+1}/{len(folds)}: val {fold['val_start']} → {fold['val_end']}")
        train, val = temporal_train_val_split(
            df,
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
        )
        yield i, train, val


# ═══════════════════════════════════════════════════════════════════════════════
# ПОДГОТОВКА МАТРИЦ X, y
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_Xy(
    df: pl.DataFrame,
    feature_cols: List[str],
    binary: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Извлекает X (фичи) и y (метки) из DataFrame.

    binary=True: y = (label == RED), игнорируем YELLOW
    binary=False: возвращаем raw label
    """
    # Фильтруем: для обучения исключаем YELLOW из таргета
    # (они не целевой класс, но могут быть в трейне с label=2)
    if binary:
        # YELLOW → 0 (не целевой); RED → 1
        y = (df[LABEL_COL] == RED).cast(pl.Int8).to_numpy()
    else:
        y = df[LABEL_COL].to_numpy()

    X = df.select(feature_cols).fill_nan(0).fill_null(0).to_numpy()
    return X, y


def prepare_Xy_no_yellow(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Убираем yellow операции из обучения полностью.
    Это даёт более чистый сигнал.
    """
    df_clean = df.filter(pl.col(LABEL_COL) != YELLOW)
    return prepare_Xy(df_clean, feature_cols, binary=True)


# ═══════════════════════════════════════════════════════════════════════════════
# МЕТРИКА
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """PR-AUC через sklearn average_precision_score (официальная метрика соревнования)."""
    if y_true.sum() == 0:
        logger.warning("No positive samples in validation set!")
        return 0.0
    return average_precision_score(y_true, y_score)


def evaluate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Дополнительные метрики при заданном пороге."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_pred = (y_score >= threshold).astype(int)
    return {
        "pr_auc":    compute_pr_auc(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "n_flagged": int(y_pred.sum()),
        "n_pos":     int(y_true.sum()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# СВОДКА ПО ФОЛДАМ
# ═══════════════════════════════════════════════════════════════════════════════

def print_cv_summary(fold_results: List[FoldResult]) -> None:
    scores = [r.pr_auc for r in fold_results]
    print(f"\n{'='*50}")
    print(f"  Cross-Validation Summary")
    print(f"{'='*50}")
    for r in fold_results:
        print(f"  Fold {r.fold_idx+1}: PR-AUC = {r.pr_auc:.4f}  "
              f"(val {r.val_start} → {r.val_end}, "
              f"pos={r.n_pos_val:,}/{r.n_val:,})")
    print(f"  {'─'*46}")
    print(f"  Mean ± Std: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"  Min / Max:  {np.min(scores):.4f} / {np.max(scores):.4f}")
    print(f"{'='*50}\n")


def build_oof_predictions(fold_results: List[FoldResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Собирает OOF предсказания всех фолдов для финального stacking."""
    all_preds  = np.concatenate([r.oof_preds  for r in fold_results])
    all_labels = np.concatenate([r.oof_labels for r in fold_results])
    oof_score  = compute_pr_auc(all_labels, all_preds)
    logger.info(f"OOF PR-AUC (all folds combined): {oof_score:.4f}")
    return all_preds, all_labels
