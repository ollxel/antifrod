"""
sequence_features.py
Sequence-based фичи: история операций как последовательность.
GRU-автоэнкодер для извлечения эмбеддингов клиентского поведения.
"""

import numpy as np
import logging
from typing import Optional
import warnings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# Попытка импортировать PyTorch
# ─────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Sequence features will be skipped.")


# ═══════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════

class TransactionSequenceDataset(Dataset):
    """
    Каждый пример = последние N операций клиента.
    Признаки каждой операции: [amount_normalized, hour, weekday, mcc_encoded, channel_encoded]
    """
    def __init__(
        self,
        sequences: np.ndarray,   # (N, seq_len, feat_dim)
        labels: Optional[np.ndarray] = None,
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


# ═══════════════════════════════════════════════
# GRU Encoder
# ═══════════════════════════════════════════════

class GRUFraudEncoder(nn.Module):
    """
    GRU энкодер истории транзакций.
    Выход: эмбеддинг + предсказание вероятности фрода.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def encode(self, x):
        """Возвращает эмбеддинг последовательности."""
        gru_out, _ = self.gru(x)              # (B, T, H)
        # Attention pooling
        attn_weights = torch.softmax(
            self.attention(gru_out), dim=1
        )                                      # (B, T, 1)
        context = (attn_weights * gru_out).sum(dim=1)  # (B, H)
        return self.encoder(context)           # (B, emb)

    def forward(self, x):
        emb = self.encode(x)
        return self.classifier(emb).squeeze(-1)


# ═══════════════════════════════════════════════
# Построение последовательностей
# ═══════════════════════════════════════════════

def build_sequences(
    df,   # polars DataFrame
    cfg: dict,
    max_len: int = 50,
    feature_cols: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Для каждой операции берём max_len предыдущих операций клиента.
    Возвращает: sequences (N, max_len, feat_dim), event_ids, client_ids
    """
    import polars as pl

    ts  = cfg["columns"]["timestamp"]
    cid = cfg["columns"]["client_id"]
    amt = cfg["columns"]["amount"]

    if feature_cols is None:
        feature_cols = [c for c in [amt, "hour", "weekday", "month"] if c in df.columns]

    logger.info(f"Building sequences: max_len={max_len}, features={feature_cols}")

    df = df.sort([cid, ts])

    # Нормализация amount
    amt_mean = df[amt].mean() or 1.0
    amt_std  = df[amt].std()  or 1.0
    df = df.with_columns(
        ((pl.col(amt) - amt_mean) / amt_std).alias("amt_norm")
    )

    feat_cols_norm = ["amt_norm"] + [c for c in feature_cols if c != amt]
    feat_dim = len(feat_cols_norm)

    event_ids  = []
    sequences  = []

    # Группируем по клиенту
    for client_data in df.partition_by(cid, maintain_order=True):
        feats = client_data.select(feat_cols_norm).to_numpy()  # (T, feat_dim)
        eids  = client_data["event_id"].to_list()

        for i, eid in enumerate(eids):
            # Берём i предыдущих операций (без текущей — look-ahead leakage!)
            start = max(0, i - max_len)
            hist  = feats[start:i]  # (<=max_len, feat_dim)

            # Паддинг нулями слева
            padded = np.zeros((max_len, feat_dim), dtype=np.float32)
            if len(hist) > 0:
                padded[-len(hist):] = hist

            sequences.append(padded)
            event_ids.append(eid)

    return np.array(sequences), event_ids


# ═══════════════════════════════════════════════
# Обучение и инференс
# ═══════════════════════════════════════════════

def train_gru_encoder(
    sequences: np.ndarray,
    labels: np.ndarray,
    cfg: dict,
) -> "GRUFraudEncoder":
    """Обучает GRU модель на последовательностях с известными метками."""
    if not TORCH_AVAILABLE:
        return None

    mc = cfg["models"]["mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training GRU on {device}, sequences: {sequences.shape}")

    dataset = TransactionSequenceDataset(sequences, labels)
    loader  = DataLoader(
        dataset,
        batch_size=mc["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    model = GRUFraudEncoder(
        input_dim=sequences.shape[2],
        hidden_dim=128,
        embedding_dim=cfg["feature_engineering"]["embedding_dim"],
        dropout=mc["dropout"],
    ).to(device)

    # Focal Loss для дисбаланса классов
    pos_weight = torch.tensor([20.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=mc["epochs"]
    )

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(mc["epochs"]):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if epoch % 5 == 0:
            logger.info(f"  Epoch {epoch:3d}/{mc['epochs']} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= mc["patience"]:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    return model


def extract_embeddings(
    model: "GRUFraudEncoder",
    sequences: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Извлекает эмбеддинги из обученного GRU."""
    if not TORCH_AVAILABLE or model is None:
        return np.zeros((len(sequences), 1))

    device = next(model.parameters()).device
    model.eval()
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.FloatTensor(sequences[i:i + batch_size]).to(device)
            emb = model.encode(batch).cpu().numpy()
            all_embs.append(emb)

    return np.vstack(all_embs)


def get_gru_predictions(
    model: "GRUFraudEncoder",
    sequences: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Предсказания GRU модели (логиты)."""
    if not TORCH_AVAILABLE or model is None:
        return np.zeros(len(sequences))

    device = next(model.parameters()).device
    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.FloatTensor(sequences[i:i + batch_size]).to(device)
            logits = model(batch).cpu().numpy()
            preds.append(logits)

    return np.concatenate(preds)
