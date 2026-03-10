"""
ensemble.py

Публичный API (ожидает main.py):
  FinalEnsemble          — класс
    .fit_weights(oof_preds, y_oof)
    .evaluate_strategies(oof_preds, y_oof)
    .predict(test_preds, strategy)  → np.ndarray
  rank_average(preds_dict)         → np.ndarray
"""

import numpy as np
import logging
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────

def _rank_norm(arr: np.ndarray) -> np.ndarray:
    """Нормировка через ранги → [0, 1]."""
    from scipy.stats import rankdata
    return rankdata(arr) / len(arr)


def rank_average(preds_dict: dict[str, np.ndarray]) -> np.ndarray:
    """
    Усредняет предсказания нескольких моделей через их ранги.
    Rank-average устойчив к масштабу и часто лучше простого среднего.
    """
    ranks = np.column_stack([_rank_norm(v) for v in preds_dict.values()])
    return ranks.mean(axis=1)


# ──────────────────────────────────────────────────────────────────
# FinalEnsemble
# ──────────────────────────────────────────────────────────────────

class FinalEnsemble:
    """
    Поддерживает три стратегии ансамблирования:
      "weighted"  — взвешенное среднее (веса оптимизированы Optuna)
      "rank"      — rank-average с равными весами
      "stacking"  — логистическая регрессия на OOF предсказаниях
    """

    def __init__(self):
        self.weights: dict[str, float] = {}
        self._stacking_scaler = None
        self._stacking_model  = None
        self._model_names: list[str] = []

    # ── Подбор весов через Optuna ──────────────────────────────────

    def fit_weights(
        self,
        oof_preds: dict[str, np.ndarray],
        y_true: np.ndarray,
        n_trials: int = 200,
        seed: int = 42,
    ) -> None:
        """Оптимизирует веса взвешенного среднего по OOF PR-AUC."""
        self._model_names = list(oof_preds.keys())

        if len(self._model_names) == 1:
            self.weights = {self._model_names[0]: 1.0}
            logger.info("Only one model — weight=1.0")
            return

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed — using equal weights")
            n = len(self._model_names)
            self.weights = {k: 1.0 / n for k in self._model_names}
            return

        # Rank-нормализуем OOF один раз
        oof_matrix = np.column_stack([
            _rank_norm(oof_preds[k]) for k in self._model_names
        ])

        def objective(trial):
            raw_w = np.array([
                trial.suggest_float(f"w_{k}", 0.0, 1.0)
                for k in self._model_names
            ])
            w = raw_w / (raw_w.sum() + 1e-9)
            blended = oof_matrix @ w
            return average_precision_score(y_true, blended)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        raw_w = np.array([study.best_params[f"w_{k}"] for k in self._model_names])
        raw_w /= raw_w.sum() + 1e-9
        self.weights = dict(zip(self._model_names, raw_w))

        logger.info(f"Optimized weights: {self.weights}")
        logger.info(f"Best OOF PR-AUC (weighted): {study.best_value:.5f}")

        # Обучаем stacking-модель на OOF
        self._fit_stacking(oof_preds, y_true)

    # ── Stacking (Level-2 логистическая регрессия) ─────────────────

    def _fit_stacking(
        self,
        oof_preds: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Обучает мета-модель на OOF предсказаниях."""
        X_meta = np.column_stack([
            _rank_norm(oof_preds[k]) for k in self._model_names
        ])
        self._stacking_scaler = StandardScaler()
        X_scaled = self._stacking_scaler.fit_transform(X_meta)

        self._stacking_model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        self._stacking_model.fit(X_scaled, y_true)

        oof_stack = self._stacking_model.predict_proba(X_scaled)[:, 1]
        stack_score = average_precision_score(y_true, oof_stack)
        logger.info(f"Stacking OOF PR-AUC: {stack_score:.5f}")

    # ── Оценка всех стратегий на OOF ──────────────────────────────

    def evaluate_strategies(
        self,
        oof_preds: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Считает PR-AUC для каждой стратегии и каждой отдельной модели."""
        results = {}

        # Отдельные модели
        for name, preds in oof_preds.items():
            score = average_precision_score(y_true, preds)
            results[name] = score
            logger.info(f"  {name:15s}: {score:.5f}")

        # Rank average
        if len(oof_preds) > 1:
            ra_score = average_precision_score(y_true, rank_average(oof_preds))
            results["rank_average"] = ra_score
            logger.info(f"  {'rank_average':15s}: {ra_score:.5f}")

        # Weighted
        if self.weights:
            wp = self._apply_weights(oof_preds)
            w_score = average_precision_score(y_true, wp)
            results["weighted"] = w_score
            logger.info(f"  {'weighted':15s}: {w_score:.5f}")

        # Stacking
        if self._stacking_model is not None:
            sp = self._apply_stacking(oof_preds)
            s_score = average_precision_score(y_true, sp)
            results["stacking"] = s_score
            logger.info(f"  {'stacking':15s}: {s_score:.5f}")

        best = max(results, key=results.get)
        logger.info(f"\n  Best strategy: {best} ({results[best]:.5f})")
        return results

    # ── Предсказание на тесте ──────────────────────────────────────

    def predict(
        self,
        test_preds: dict[str, np.ndarray],
        strategy: str = "weighted",
    ) -> np.ndarray:
        """
        Применяет выбранную стратегию к тестовым предсказаниям.
        strategy: "weighted" | "rank" | "stacking"
        """
        if strategy == "rank":
            return rank_average(test_preds)

        elif strategy == "weighted":
            if not self.weights:
                logger.warning("Weights not fitted, falling back to rank_average")
                return rank_average(test_preds)
            return self._apply_weights(test_preds)

        elif strategy == "stacking":
            if self._stacking_model is None:
                logger.warning("Stacking not fitted, falling back to rank_average")
                return rank_average(test_preds)
            return self._apply_stacking(test_preds)

        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use: weighted | rank | stacking")

    def _apply_weights(self, preds_dict: dict[str, np.ndarray]) -> np.ndarray:
        names = [k for k in self._model_names if k in preds_dict]
        matrix = np.column_stack([_rank_norm(preds_dict[k]) for k in names])
        w = np.array([self.weights.get(k, 0.0) for k in names])
        w = w / (w.sum() + 1e-9)
        return matrix @ w

    def _apply_stacking(self, preds_dict: dict[str, np.ndarray]) -> np.ndarray:
        names  = [k for k in self._model_names if k in preds_dict]
        X_meta = np.column_stack([_rank_norm(preds_dict[k]) for k in names])
        X_s    = self._stacking_scaler.transform(X_meta)
        return self._stacking_model.predict_proba(X_s)[:, 1]