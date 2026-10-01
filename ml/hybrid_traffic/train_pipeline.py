"""
Phase 5 Training Pipeline
=========================

Complete training, validation, and evaluation pipeline for the Proposed
TCN-Transformer Gated Hybrid model on the 9-target vehicle trajectory &
traffic intelligence dataset.

Strict requirements:
1. Zero modification of Phase 4 datasets.
2. Standard scaling fitted strictly on train split (zero leakage).
3. Early stopping and model selection evaluated exclusively on validation split.
4. Untouched test split evaluated exactly ONCE on best saved checkpoint.
5. Full generation of metrics, checkpoints, plots, logs, and prediction files.
"""

import os
import sys
import time
import json
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    confusion_matrix
)
import joblib
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from models.tcn_transformer_hybrid import TCNTransformerHybrid, MultiTaskLoss


def set_seed(seed: int = 42) -> None:
    """Set random seeds across libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrafficDataset(Dataset):
    """PyTorch Dataset wrapper for sequence tensors and 9 multi-task targets."""

    def __init__(self, X: np.ndarray, labels_dict: Dict[str, np.ndarray], track_ids: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.track_ids = track_ids
        self.targets = {}

        # 9 Target columns + auxiliary
        int_targets = [
            "target_congestion_level",
            "target_risk_level",
            "target_is_approaching",
            "target_motion_state",
            "target_maneuver_type",
            "target_has_infraction",
            "target_infraction_type"
        ]
        float_targets = [
            "target_congestion_score",
            "target_approach_score",
            "aux_next_displacement"
        ]

        for k in int_targets:
            if k in labels_dict:
                self.targets[k] = torch.from_numpy(labels_dict[k].astype(np.int64))

        for k in float_targets:
            if k in labels_dict:
                self.targets[k] = torch.from_numpy(labels_dict[k].astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        item_targets = {k: v[idx] for k, v in self.targets.items()}
        return self.X[idx], item_targets


def load_raw_data() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Loads Phase 3 and Phase 4 NPZ files in strict read-only mode."""
    seq_dir = "data/hybrid_traffic/sequences"
    lbl_dir = "data/hybrid_traffic/labels"

    # Read-only loading
    train_seq = np.load(os.path.join(seq_dir, "train_sequences.npz"), mmap_mode="r")
    train_lbl = np.load(os.path.join(lbl_dir, "train_labels.npz"), mmap_mode="r")

    val_seq = np.load(os.path.join(seq_dir, "val_sequences.npz"), mmap_mode="r")
    val_lbl = np.load(os.path.join(lbl_dir, "val_labels.npz"), mmap_mode="r")

    test_seq = np.load(os.path.join(seq_dir, "test_sequences.npz"), mmap_mode="r")
    test_lbl = np.load(os.path.join(lbl_dir, "test_labels.npz"), mmap_mode="r")

    train_data = {
        "X": np.array(train_seq["X"], dtype=np.float32),
        "track_ids": np.array(train_seq["track_ids"]),
        "aux_next_displacement": np.nan_to_num(np.array(train_seq["y_next_disp"], dtype=np.float32), nan=0.0)
    }
    for k in train_lbl.files:
        if k.startswith("target_"):
            train_data[k] = np.array(train_lbl[k])

    val_data = {
        "X": np.array(val_seq["X"], dtype=np.float32),
        "track_ids": np.array(val_seq["track_ids"]),
        "aux_next_displacement": np.nan_to_num(np.array(val_seq["y_next_disp"], dtype=np.float32), nan=0.0)
    }
    for k in val_lbl.files:
        if k.startswith("target_"):
            val_data[k] = np.array(val_lbl[k])

    test_data = {
        "X": np.array(test_seq["X"], dtype=np.float32),
        "track_ids": np.array(test_seq["track_ids"]),
        "aux_next_displacement": np.nan_to_num(np.array(test_seq["y_next_disp"], dtype=np.float32), nan=0.0)
    }
    for k in test_lbl.files:
        if k.startswith("target_"):
            test_data[k] = np.array(test_lbl[k])

    return train_data, val_data, test_data


def apply_feature_scaling(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    test_data: Dict[str, Any],
    scaler_save_paths: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler EXCLUSIVELY on the training sequences and transforms
    train, val, and test splits to guarantee zero data leakage.
    """
    X_train = train_data["X"]
    X_val = val_data["X"]
    X_test = test_data["X"]

    n_train, t_len, n_feat = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    # Flatten temporal dimension for feature-wise scaling
    X_train_flat = X_train.reshape(-1, n_feat)
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    # Transform all splits
    X_train_scaled = scaler.transform(X_train_flat).reshape(n_train, t_len, n_feat).astype(np.float32)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_feat)).reshape(n_val, t_len, n_feat).astype(np.float32)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_feat)).reshape(n_test, t_len, n_feat).astype(np.float32)

    # Save scaler artifact
    for p in scaler_save_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        joblib.dump(scaler, p)
        print(f"[PREPROCESSING] Saved fitted StandardScaler to: {p}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def calculate_class_weights(train_labels: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Computes inverse class frequency weights from the training set."""
    weights = {}

    # target_risk_level
    risk_counts = np.bincount(train_labels["target_risk_level"], minlength=3)
    w_risk = len(train_labels["target_risk_level"]) / (3.0 * np.maximum(risk_counts, 1))
    weights["target_risk_level"] = torch.tensor(w_risk, dtype=torch.float32)

    # target_motion_state
    motion_counts = np.bincount(train_labels["target_motion_state"], minlength=4)
    w_motion = len(train_labels["target_motion_state"]) / (4.0 * np.maximum(motion_counts, 1))
    weights["target_motion_state"] = torch.tensor(w_motion, dtype=torch.float32)

    # target_maneuver_type
    maneuver_counts = np.bincount(train_labels["target_maneuver_type"], minlength=4)
    w_man = len(train_labels["target_maneuver_type"]) / (4.0 * np.maximum(maneuver_counts, 1))
    weights["target_maneuver_type"] = torch.tensor(w_man, dtype=torch.float32)

    # target_infraction_type (Class 2 has 0 samples, assign weight 0 to ignore in loss)
    inf_counts = np.bincount(train_labels["target_infraction_type"], minlength=3)
    w_inf = np.zeros(3, dtype=np.float32)
    valid_classes = inf_counts > 0
    w_inf[valid_classes] = len(train_labels["target_infraction_type"]) / (2.0 * inf_counts[valid_classes])
    weights["target_infraction_type"] = torch.tensor(w_inf, dtype=torch.float32)

    return weights


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """Runs one training epoch."""
    model.train()
    total_loss_accum = 0.0
    task_loss_accum = {k: 0.0 for k in criterion.task_names}
    num_batches = len(dataloader)

    for batch_X, batch_targets in dataloader:
        batch_X = batch_X.to(device)
        targets = {k: v.to(device) for k, v in batch_targets.items()}

        optimizer.zero_grad()
        predictions = model(batch_X)
        total_loss, ind_losses = criterion(predictions, targets)

        total_loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss_accum += total_loss.item()
        for k in criterion.task_names:
            if k in ind_losses:
                task_loss_accum[k] += ind_losses[k].item()

    avg_total_loss = total_loss_accum / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_loss_accum.items()}
    return avg_total_loss, avg_task_losses


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device
) -> Tuple[float, Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Evaluates model over a dataloader without computing gradients."""
    model.eval()
    total_loss_accum = 0.0
    task_loss_accum = {k: 0.0 for k in criterion.task_names}
    num_batches = len(dataloader)

    all_preds = {k: [] for k in criterion.task_names}
    all_targets = {k: [] for k in criterion.task_names}

    with torch.no_grad():
        for batch_X, batch_targets in dataloader:
            batch_X = batch_X.to(device)
            targets = {k: v.to(device) for k, v in batch_targets.items()}

            predictions = model(batch_X)
            total_loss, ind_losses = criterion(predictions, targets)

            total_loss_accum += total_loss.item()
            for k in criterion.task_names:
                if k in ind_losses:
                    task_loss_accum[k] += ind_losses[k].item()
                if k in predictions:
                    all_preds[k].append(predictions[k].cpu().numpy())
                if k in targets:
                    all_targets[k].append(targets[k].cpu().numpy())

    avg_total_loss = total_loss_accum / num_batches
    avg_task_losses = {k: v / num_batches for k, v in task_loss_accum.items()}

    concatenated_preds = {k: np.concatenate(v, axis=0) for k, v in all_preds.items() if len(v) > 0}
    concatenated_targets = {k: np.concatenate(v, axis=0) for k, v in all_targets.items() if len(v) > 0}

    return avg_total_loss, avg_task_losses, concatenated_preds, concatenated_targets


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    task_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Computes detailed statistical and domain evaluation metrics per target."""
    metrics_report = {}

    for name, cfg in task_configs.items():
        if name not in predictions or name not in targets:
            continue

        pred = predictions[name]
        y_true = targets[name]
        t_type = cfg.get("type", "multiclass")

        if t_type in ("multiclass", "binary"):
            # Probabilities / logits to class predictions
            if pred.ndim > 1:
                y_pred = np.argmax(pred, axis=1)
                probs = torch.softmax(torch.from_numpy(pred), dim=-1).numpy()
            else:
                y_pred = (pred > 0.5).astype(int)
                probs = pred

            acc = accuracy_score(y_true, y_pred)
            prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            cm = confusion_matrix(y_true, y_pred).tolist()

            target_m = {
                "type": t_type,
                "accuracy": float(acc),
                "precision_macro": float(prec_macro),
                "recall_macro": float(rec_macro),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "confusion_matrix": cm
            }

            # Binary ROC-AUC
            if t_type == "binary" and pred.ndim > 1 and pred.shape[1] == 2:
                try:
                    auc = roc_auc_score(y_true, probs[:, 1])
                    target_m["roc_auc"] = float(auc)
                except Exception:
                    pass

            metrics_report[name] = target_m

        elif t_type == "regression":
            # Regression metrics
            mae = mean_absolute_error(y_true, pred)
            mse = mean_squared_error(y_true, pred)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_true, pred)

            target_m = {
                "type": "regression",
                "mae": float(mae),
                "mse": float(mse),
                "rmse": float(rmse),
                "r2": float(r2)
            }

            # Safe MAPE calculation for target distributions strictly away from zero
            # e.g., target_congestion_score is in [10, 100]
            if name == "target_congestion_score" and np.all(y_true > 0):
                mape = float(np.mean(np.abs((y_true - pred) / y_true)) * 100.0)
                target_m["mape_pct"] = mape

            metrics_report[name] = target_m

    return metrics_report


def generate_training_plots(history: Dict[str, List[float]], output_dir: str) -> None:
    """Generates and saves publication-quality training curve plots."""
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Total Training vs Validation Loss Curve
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], label="Train Total Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Total Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Phase 5: Hybrid Model Total Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Multi-Task Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=200)
    plt.close()

    # 2. Validation Loss Specifically
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["val_loss"], label="Validation Loss", color="#2ca02c", linewidth=2)
    best_ep = np.argmin(history["val_loss"]) + 1
    best_val = min(history["val_loss"])
    plt.scatter([best_ep], [best_val], color="red", s=100, zorder=5, label=f"Best Model (Epoch {best_ep}: {best_val:.4f})")
    plt.title("Phase 5: Validation Loss & Best Model Selection", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_loss.png"), dpi=200)
    plt.close()

    # 3. Learning Rate Curve
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, history["learning_rate"], label="Learning Rate", color="#d62728", linewidth=2)
    plt.title("Learning Rate Schedule (ReduceLROnPlateau / Cosine)", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_rate.png"), dpi=200)
    plt.close()

    # 4. Multi-Task Loss Breakdown (Individual Targets)
    plt.figure(figsize=(11, 6))
    for t_name, loss_vals in history["task_val_losses"].items():
        if len(loss_vals) == len(epochs):
            plt.plot(epochs, loss_vals, label=t_name.replace("target_", ""), linewidth=1.5)
    plt.title("Validation Loss Breakdown by Individual Target", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Task Loss", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multi_task_losses.png"), dpi=200)
    plt.close()

    print(f"[PLOTS] Generated training_loss.png, validation_loss.png, learning_rate.png, multi_task_losses.png in {output_dir}")


def generate_confusion_matrix_plots(test_metrics: Dict[str, Dict[str, Any]], output_dir: str) -> None:
    """Generates visual confusion matrix grids for classification targets."""
    class_targets = [k for k, v in test_metrics.items() if v.get("type") in ("multiclass", "binary") and "confusion_matrix" in v]
    if not class_targets:
        return

    n_targets = len(class_targets)
    cols = 3
    rows = (n_targets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, t_name in enumerate(class_targets):
        ax = axes[idx]
        cm = np.array(test_metrics[t_name]["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(t_name.replace("target_", ""), fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate numbers
        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

        ax.set_ylabel("True Label", fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=9)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"), dpi=200)
    plt.close()
    print(f"[PLOTS] Generated confusion_matrices.png in {output_dir}")


def run_pipeline(eval_only: bool = False) -> None:
    """Executes the complete Phase 5 training, validation, and evaluation pipeline."""
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    start_time = time.time()
    set_seed(42)

    # 1. Directories setup
    dirs = ["checkpoints", "logs", "metrics", "predictions", "plots", "data/hybrid_traffic/scalers"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("=" * 80)
    print("PHASE 5: TRAINING PIPELINE FOR TCN-TRANSFORMER GATED HYBRID MODEL")
    print("=" * 80)

    # 2. Load Configuration
    config_path = "configs/phase5_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 3. Load Raw Data
    print("\n[DATA] Loading Phase 3 sequences and Phase 4 labels...")
    train_data, val_data, test_data = load_raw_data()

    print(f"  Train sequences:      {train_data['X'].shape[0]:,}")
    print(f"  Validation sequences: {val_data['X'].shape[0]:,}")
    print(f"  Test sequences:       {test_data['X'].shape[0]:,}")

    # 4. Apply Scaling
    scaler_paths = [
        "checkpoints/feature_scaler.joblib",
        "data/hybrid_traffic/scalers/feature_scaler.joblib"
    ]
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_feature_scaling(
        train_data, val_data, test_data, scaler_paths
    )

    # 5. Build PyTorch Datasets & DataLoaders
    batch_size = config["training"].get("batch_size", 64)
    train_dataset = TrafficDataset(X_train_scaled, train_data, train_data["track_ids"])
    val_dataset = TrafficDataset(X_val_scaled, val_data, val_data["track_ids"])
    test_dataset = TrafficDataset(X_test_scaled, test_data, test_data["track_ids"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Instantiate Architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Compute device: {device} (AMD CPU Execution)")

    m_cfg = config["model"]
    model = TCNTransformerHybrid(
        input_dim=config["dataset"]["feature_count"],
        sequence_length=config["dataset"]["sequence_length"],
        hidden_dim=m_cfg.get("hidden_dim", 64),
        tcn_kernel_size=m_cfg["tcn"].get("kernel_size", 3),
        tcn_dilation_rates=m_cfg["tcn"].get("dilation_rates", [1, 2, 4, 8]),
        tcn_channels=m_cfg["tcn"].get("channels", [64, 64, 64, 64]),
        transformer_layers=m_cfg["transformer"].get("num_layers", 2),
        transformer_heads=m_cfg["transformer"].get("num_heads", 4),
        transformer_ff_dim=m_cfg["transformer"].get("feedforward_dim", 256),
        dropout=m_cfg.get("dropout", 0.2)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] TCN-Transformer Gated Hybrid instantiated. Total Trainable Parameters: {total_params:,}")

    # 7. Loss Configuration
    class_weights = calculate_class_weights(train_data)
    task_configs = config["dataset"]["targets"]

    # Calibrate task weights for loss scale parity
    calibrated_task_configs = {
        "target_congestion_level": {"type": "multiclass", "weight": 1.0, "loss": "cross_entropy"},
        "target_congestion_score": {"type": "regression", "weight": 0.05, "loss": "smooth_l1"},
        "target_risk_level": {"type": "multiclass", "weight": 1.2, "loss": "cross_entropy"},
        "target_is_approaching": {"type": "binary", "weight": 1.0, "loss": "cross_entropy"},
        "target_approach_score": {"type": "regression", "weight": 0.05, "loss": "smooth_l1"},
        "target_motion_state": {"type": "multiclass", "weight": 1.0, "loss": "cross_entropy"},
        "target_maneuver_type": {"type": "multiclass", "weight": 1.0, "loss": "cross_entropy"},
        "target_has_infraction": {"type": "binary", "weight": 1.2, "loss": "cross_entropy"},
        "target_infraction_type": {"type": "multiclass", "weight": 0.8, "loss": "cross_entropy"},
        "aux_next_displacement": {"type": "regression", "weight": 0.5, "loss": "smooth_l1"}
    }

    criterion = MultiTaskLoss(
        task_configs=calibrated_task_configs,
        weighting_mode="static",
        class_weights={k: v.to(device) for k, v in class_weights.items()}
    ).to(device)

    # 8. Optimizer & Scheduler
    lr = float(config["training"].get("learning_rate", 0.001))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    # 9. Training Loop
    epochs = int(config["training"].get("epochs", 30))
    patience = int(config["training"]["early_stopping"].get("patience", 7))
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history: Dict[str, Any] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "task_train_losses": {k: [] for k in criterion.task_names},
        "task_val_losses": {k: [] for k in criterion.task_names}
    }

    if not eval_only:
        print("\n" + "-" * 80)
        print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'LR':<10} | {'Status'}")
        print("-" * 80)

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            current_lr = optimizer.param_groups[0]["lr"]

            # Train & Evaluate
            train_loss, train_tasks = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip_norm=1.0
            )
            val_loss, val_tasks, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            scheduler.step(val_loss)
            ep_duration = time.time() - ep_start

            # Record History
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["learning_rate"].append(current_lr)
            for k in criterion.task_names:
                history["task_train_losses"][k].append(train_tasks.get(k, 0.0))
                history["task_val_losses"][k].append(val_tasks.get(k, 0.0))

            # Checkpointing
            is_best = val_loss < best_val_loss
            status_msg = f"{ep_duration:.1f}s"
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                status_msg += " [BEST CHECKPOINT SAVED]"

                checkpoint_data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "config": config,
                    "calibrated_task_configs": calibrated_task_configs,
                    "total_params": total_params
                }
                torch.save(checkpoint_data, "checkpoints/best_model.pth")
            else:
                patience_counter += 1
                status_msg += f" (Patience {patience_counter}/{patience})"

            print(f"{epoch:<6} | {train_loss:<12.4f} | {val_loss:<12.4f} | {current_lr:<10.2e} | {status_msg}")

            # Early stopping trigger
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation loss stopped improving for {patience} consecutive epochs. Terminating training.")
                break

        # Save training logs and history
        with open("logs/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        log_df = pd.DataFrame({
            "epoch": history["epoch"],
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "learning_rate": history["learning_rate"]
        })
        log_df.to_csv("logs/training_log.csv", index=False)

        # Save exact configuration used
        with open("checkpoints/best_model_config.json", "w") as f:
            json.dump({
                "model_architecture": "TCNTransformerHybrid",
                "total_params": total_params,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "config": config,
                "calibrated_task_configs": calibrated_task_configs
            }, f, indent=2)

        # Generate plots
        generate_training_plots(history, "plots")
    else:
        best_ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=False)
        best_epoch = best_ckpt.get("epoch", 7)
        best_val_loss = best_ckpt.get("val_loss", 6.6147)

    # =========================================================================
    # 10. FINAL TEST SET EVALUATION (EXACTLY ONCE ON BEST MODEL)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL TEST SET EVALUATION (UNTOUCHED TEST SPLIT: 2,597 SEQUENCES)")
    print("=" * 80)

    # Load best checkpoint
    print(f"[EVALUATION] Loading best model checkpoint from Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})...")
    best_ckpt = torch.load("checkpoints/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_total_loss, test_task_losses, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    print(f"[TEST EVALUATION] Test Total Loss: {test_total_loss:.4f}")

    # Compute per-target metrics
    test_metrics = compute_metrics(test_preds, test_targets, calibrated_task_configs)

    # Save predictions
    save_preds = {k: v for k, v in test_preds.items()}
    save_preds["track_ids"] = test_data["track_ids"]
    np.savez_compressed("predictions/test_predictions.npz", **save_preds)

    # Build predictions summary CSV
    summary_dict = {"track_id": test_data["track_ids"]}
    for k, v in test_preds.items():
        if k == "aux_next_displacement":
            summary_dict["pred_aux_dx"] = v[:, 0]
            summary_dict["pred_aux_dy"] = v[:, 1]
            summary_dict["true_aux_dx"] = test_targets[k][:, 0]
            summary_dict["true_aux_dy"] = test_targets[k][:, 1]
        elif v.ndim == 2:
            summary_dict[f"pred_{k}_class"] = np.argmax(v, axis=1)
            summary_dict[f"true_{k}_class"] = test_targets[k]
        else:
            summary_dict[f"pred_{k}"] = v
            summary_dict[f"true_{k}"] = test_targets[k]

    pd.DataFrame(summary_dict).to_csv("predictions/test_predictions_summary.csv", index=False)

    # Save metrics JSON
    with open("metrics/final_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save per-target summary table
    per_target_rows = []
    for k, m in test_metrics.items():
        row = {"target_name": k, "type": m["type"]}
        if m["type"] in ("multiclass", "binary"):
            row["accuracy"] = m["accuracy"]
            row["precision_macro"] = m["precision_macro"]
            row["recall_macro"] = m["recall_macro"]
            row["f1_macro"] = m["f1_macro"]
            row["f1_weighted"] = m["f1_weighted"]
            row["roc_auc"] = m.get("roc_auc", None)
            row["mae"] = None
            row["rmse"] = None
            row["r2"] = None
        else:
            row["accuracy"] = None
            row["precision_macro"] = None
            row["recall_macro"] = None
            row["f1_macro"] = None
            row["f1_weighted"] = None
            row["roc_auc"] = None
            row["mae"] = m["mae"]
            row["rmse"] = m["rmse"]
            row["r2"] = m["r2"]
        per_target_rows.append(row)

    pd.DataFrame(per_target_rows).to_csv("metrics/per_target_metrics.csv", index=False)
    generate_confusion_matrix_plots(test_metrics, "plots")

    total_training_time = time.time() - start_time
    minutes = int(total_training_time // 60)
    seconds = int(total_training_time % 60)
    time_str = f"{minutes}m {seconds}s"

    print("\n" + "=" * 80)
    print(f"MODEL: TCN-Transformer Gated Hybrid")
    print(f"TRAIN: {len(train_dataset)} sequences")
    print(f"VALIDATION: {len(val_dataset)} sequences")
    print(f"TEST: {len(test_dataset)} sequences")
    print(f"TARGETS: 9 (+1 auxiliary trajectory head)")
    print(f"BEST EPOCH: {best_epoch}")
    print(f"BEST VALIDATION LOSS: {best_val_loss:.4f}")
    print(f"TEST LOSS: {test_total_loss:.4f}")
    print(f"MODEL PARAMETERS: {total_params:,}")
    print(f"TRAINING TIME: {time_str}")
    print("=" * 80)

    print("\nTEST PERFORMANCE SUMMARY:")
    for k, m in test_metrics.items():
        if m["type"] in ("multiclass", "binary"):
            print(f"  * {k:<25}: Acc={m['accuracy']:.4f} | Macro-F1={m['f1_macro']:.4f} | Weighted-F1={m['f1_weighted']:.4f}")
        else:
            print(f"  * {k:<25}: MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R²={m['r2']:.4f}")


if __name__ == "__main__":
    eval_only_flag = "--eval-only" in sys.argv
    run_pipeline(eval_only=eval_only_flag)
