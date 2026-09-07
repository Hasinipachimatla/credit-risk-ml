"""
PyTorch Deep Learning Tabular Neural Network for CreditRiskML.
Implements a multi-layer feedforward network with BatchNorm, Dropout,
weighted BCEWithLogitsLoss, early stopping, and checkpointing.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger("credit_risk_ml.pytorch")

class TabularCreditDataset(Dataset):
    """PyTorch Dataset for tabular credit features and binary targets."""
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class CreditRiskNet(nn.Module):
    """
    Tabular Feedforward Neural Network for credit default prediction.
    Uses BatchNorm and Dropout for regularizing high-dimensional encoded tabular features.
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.2):
        super(CreditRiskNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim2, 1)  # Raw logits output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def set_seed(seed: int = 42):
    """Sets deterministic random seeds."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_pytorch_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
    checkpoint_path: Optional[Path] = None
) -> Tuple[CreditRiskNet, Dict[str, Any]]:
    """
    Trains CreditRiskNet using weighted BCEWithLogitsLoss and early stopping on validation loss.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = CreditRiskNet(input_dim=input_dim).to(device)

    # Calculate positive class weight to counter 70/30 class imbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    train_dataset = TabularCreditDataset(X_train, y_train)
    val_dataset = TabularCreditDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            if checkpoint_path:
                torch.save({
                    "model_state_dict": best_model_state,
                    "input_dim": input_dim,
                    "epoch": epoch,
                    "val_loss": best_val_loss
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    logger.info(f"PyTorch training completed. Best Val Loss: {best_val_loss:.4f}")
    return model, history

def predict_proba_pytorch(model: CreditRiskNet, X: np.ndarray) -> np.ndarray:
    """Computes default probabilities using sigmoid on network logits."""
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs
