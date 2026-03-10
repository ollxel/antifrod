"""
utils.py — Вспомогательные функции.
"""

import yaml
import logging
import numpy as np
import random
import os
from pathlib import Path
from datetime import datetime


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timer(name: str):
    """Context manager для замера времени."""
    import time
    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            logging.getLogger(__name__).info(
                f"[{name}] done in {elapsed:.1f}s"
            )
    return Timer()


def save_predictions(
    event_ids: list,
    predictions: np.ndarray,
    output_path: str,
):
    """Сохраняет сабмит в формате соревнования."""
    import polars as pl
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        "event_id": event_ids,
        "predict": predictions.astype(float),
    })
    df.write_csv(output_path)
    logging.getLogger(__name__).info(
        f"Submission saved: {output_path} ({len(df):,} rows)"
    )
    return df


def memory_usage_mb() -> float:
    import psutil, os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def encode_categoricals(df, cat_cols: list):
    """Label-encoding категориальных колонок для деревьев."""
    import polars as pl
    for col in cat_cols:
        if col in df.columns and df[col].dtype == pl.Utf8:
            mapping = {v: i for i, v in enumerate(df[col].unique().sort().to_list())}
            df = df.with_columns(
                pl.col(col).replace(mapping).cast(pl.Int32).alias(col)
            )
    return df
