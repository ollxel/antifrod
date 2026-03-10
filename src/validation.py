"""
validation.py

Публичный API (ожидает main.py):
  compute_pr_auc(y_true, y_score)  → float
  build_oof_predictions(fold_results) → (oof_preds, oof_labels)
"""

import numpy as np
import logging
from sklearn.metrics import average_precision_score, precision_recall_curve

logger = logging.getLogger(__name__)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    PR-AUC через sklearn average_precision_score.
    Это официальная метрика соревнования.
    """
    if y_true.sum() == 0:
        logger.warning("No positive labels in y_true, PR-AUC=0")
        return 0.0
    return float(average_precision_score(y_true, y_score))


def build_oof_predictions(fold_results: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Собирает OOF предсказания из результатов всех фолдов.

    fold_results — список dict:
      {"oof_preds": np.ndarray, "oof_labels": np.ndarray, ...}

    Возвращает (oof_preds, oof_labels) — конкатенация по всем фолдам.
    """
    if not fold_results:
        return np.array([]), np.array([])

    all_preds  = np.concatenate([r["oof_preds"]  for r in fold_results])
    all_labels = np.concatenate([r["oof_labels"] for r in fold_results])

    score = compute_pr_auc(all_labels, all_preds)
    logger.info(f"OOF PR-AUC (all folds): {score:.5f}")

    return all_preds, all_labels


def print_fold_summary(fold_results: list, model_name: str = "Model") -> None:
    """Логирует итоги по фолдам."""
    if not fold_results:
        return
    scores = [r["pr_auc"] for r in fold_results]
    logger.info(f"\n{model_name} Fold Summary:")
    for r in fold_results:
        logger.info(f"  Fold {r['fold']+1}: PR-AUC = {r['pr_auc']:.5f}")
    logger.info(f"  Mean: {np.mean(scores):.5f} ± {np.std(scores):.5f}")


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Находит порог, максимизирующий F1.
    Полезно для бинарных предсказаний (не для ранжирования).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1[:-1])
    best_thr = thresholds[best_idx]
    logger.info(f"Optimal threshold: {best_thr:.4f} | "
                f"P={precision[best_idx]:.4f} | R={recall[best_idx]:.4f} | F1={f1[best_idx]:.4f}")
    return float(best_thr)