"""
stacking.py
Ансамблирование через OOF-stacking и оптимизацию весов через Optuna.
"""

import numpy as np
import logging
from typing import Optional
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────

def rank_normalize(preds: np.ndarray) -> np.ndarray:
    """Нормализует предсказания через ранги → [0, 1]."""
    from scipy.stats import rankdata
    return rankdata(preds) / len(preds)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# ──────────────────────────────────────────
# Rank Average (простой и сильный baseline)
# ──────────────────────────────────────────

def rank_average(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """
    Усредняет предсказания разных моделей через их ранги.
    Rank average устойчив к масштабу и часто лучше простого усреднения.
    """
    ranks = np.column_stack([rank_normalize(p) for p in predictions.values()])
    return ranks.mean(axis=1)


# ──────────────────────────────────────────
# Optuna-оптимизация весов
# ──────────────────────────────────────────

def optimize_blend_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    n_trials: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """
    Подбирает оптимальные веса для взвешенного усреднения
    через максимизацию OOF PR-AUC.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not available. Using equal weights.")
        n = len(oof_preds)
        return {k: 1.0 / n for k in oof_preds}

    model_names = list(oof_preds.keys())
    preds_matrix = np.column_stack([rank_normalize(oof_preds[k]) for k in model_names])

    def objective(trial):
        weights = np.array([
            trial.suggest_float(f"w_{name}", 0.0, 1.0)
            for name in model_names
        ])
        weights /= weights.sum() + 1e-9
        blended = preds_matrix @ weights
        return average_precision_score(y_true, blended)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    weights = {name: best[f"w_{name}"] for name in model_names}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    logger.info(f"Best blend weights: {weights}")
    logger.info(f"Best OOF PR-AUC: {study.best_value:.5f}")

    return weights


# ──────────────────────────────────────────
# Logistic meta-model (Level-2)
# ──────────────────────────────────────────

def logistic_stacking(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    test_preds: dict[str, np.ndarray],
    cv_splits: int = 5,
) -> np.ndarray:
    """
    Уровень 2: логистическая регрессия на OOF предсказаниях.
    Использует кросс-валидацию для предотвращения переобучения мета-модели.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    model_names = list(oof_preds.keys())
    X_meta = np.column_stack([rank_normalize(oof_preds[k]) for k in model_names])
    X_test = np.column_stack([rank_normalize(test_preds[k]) for k in model_names])

    meta_oof = np.zeros(len(y_true))
    meta_test = np.zeros(len(X_test))

    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_meta, y_true)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_meta[tr_idx])
        X_val = scaler.transform(X_meta[val_idx])
        X_te  = scaler.transform(X_test)

        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(X_tr, y_true[tr_idx])

        meta_oof[val_idx] = meta.predict_proba(X_val)[:, 1]
        meta_test += meta.predict_proba(X_te)[:, 1] / cv_splits

    oof_score = average_precision_score(y_true, meta_oof)
    logger.info(f"Logistic stacking OOF PR-AUC: {oof_score:.5f}")

    return meta_test


# ──────────────────────────────────────────
# Главная функция ансамблирования
# ──────────────────────────────────────────

def ensemble(
    oof_preds: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """
    Выбирает метод ансамблирования из конфига и применяет его.
    """
    method = cfg["stacking"]["method"]
    logger.info(f"Ensembling with method: {method}")

    if method == "rank_average":
        return rank_average(test_preds)

    elif method == "optuna_weights":
        weights = optimize_blend_weights(
            oof_preds,
            y_true,
            n_trials=cfg["stacking"]["weights_optuna_trials"],
            seed=cfg["random_seed"],
        )
        model_names = list(test_preds.keys())
        preds_matrix = np.column_stack([
            rank_normalize(test_preds[k]) for k in model_names
        ])
        weight_vector = np.array([weights[k] for k in model_names])
        return preds_matrix @ weight_vector

    elif method == "logistic":
        return logistic_stacking(oof_preds, y_true, test_preds)

    else:
        raise ValueError(f"Unknown stacking method: {method}")
