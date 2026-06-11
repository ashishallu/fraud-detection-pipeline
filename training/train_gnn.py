"""
training/train_gnn.py
Trains the HeteroFraudGNN on the pre-built graph.

Features:
  - Stratified train/val/test split (70/15/15)
  - Weighted cross-entropy for class imbalance
  - AUC-ROC, precision, recall, F1 tracking per epoch
  - Best model checkpoint saved on val AUC
  - Training curves saved as PNG
"""

import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from graph.graph_builder import load_graph
from graph.gnn_model import build_model, save_model

log = logging.getLogger("train_gnn")


def create_masks(data: HeteroData) -> tuple:
    """
    Create stratified train/val/test masks on transaction nodes.

    Returns:
        (train_mask, val_mask, test_mask) — boolean tensors of shape [N_transactions]
    """
    n = data["transaction"].y.shape[0]
    labels = data["transaction"].y.numpy()
    indices = np.arange(n)

    # Stratified split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1 - config.GNN_TRAIN_RATIO),
        stratify=labels, random_state=42
    )
    # Split temp into val and test (50/50 of the 30%)
    val_ratio_of_temp = config.GNN_VAL_RATIO / (config.GNN_VAL_RATIO + config.GNN_TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_ratio_of_temp),
        stratify=labels[temp_idx], random_state=42
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    log.info(
        "Split — train: %d, val: %d, test: %d",
        train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()
    )
    fraud_in_train = data["transaction"].y[train_mask].sum().item()
    log.info("Fraud in train set: %d / %d", fraud_in_train, train_mask.sum().item())

    return train_mask, val_mask, test_mask


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute fraud/non-fraud class weights to handle imbalance.
    Weight for fraud class = (1 - fraud_rate) / fraud_rate
    """
    fraud_rate = labels.float().mean().item()
    fraud_weight = (1 - fraud_rate) / max(fraud_rate, 1e-6)
    log.info("Fraud rate: %.4f — applying fraud class weight: %.2f", fraud_rate, fraud_weight)
    pos_weight = torch.tensor([fraud_weight], dtype=torch.float)
    return pos_weight


def evaluate(
    model: nn.Module,
    data: HeteroData,
    mask: torch.Tensor,
    device: torch.device,
    threshold: float = config.FRAUD_THRESHOLD,
) -> dict:
    """
    Evaluate the model on a given mask split.

    Returns:
        dict with loss, auc, precision, recall, f1
    """
    model.eval()
    with torch.no_grad():
        probs = model(data.x_dict, data.edge_index_dict)
        labels = data["transaction"].y.float().to(device)

        pos_weight = compute_class_weights(data["transaction"].y[mask])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

        # Re-run to get logits for loss (sigmoid was applied in forward)
        # Use raw probs (already sigmoided) for metrics
        probs_mask = probs[mask].cpu().numpy()
        labels_mask = labels[mask].cpu().numpy()

        preds = (probs_mask >= threshold).astype(int)
        try:
            auc = roc_auc_score(labels_mask, probs_mask)
        except ValueError:
            auc = 0.5

        precision = precision_score(labels_mask, preds, zero_division=0)
        recall = recall_score(labels_mask, preds, zero_division=0)
        f1 = f1_score(labels_mask, preds, zero_division=0)

        loss = float(nn.BCELoss()(
            torch.tensor(probs_mask), torch.tensor(labels_mask)
        ))

    return {
        "loss": loss,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probs": probs_mask,
        "labels": labels_mask,
        "preds": preds,
    }


def plot_training_curves(history: dict, save_path: Path) -> None:
    """Plot and save loss + AUC training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(epochs, history["val_loss"], label="Val Loss", color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["val_auc"], label="Val AUC-ROC", color="#4CAF50")
    ax.axhline(y=max(history["val_auc"]), color="gray", linestyle="--",
               label=f"Best AUC: {max(history['val_auc']):.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Validation AUC-ROC")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Training curves saved to %s", save_path)


def train(
    model: nn.Module,
    data: HeteroData,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Main training loop.

    Returns:
        history dict with per-epoch metrics
    """
    model = model.to(device)

    # Move graph data to device
    data = data.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.GNN_LEARNING_RATE,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    # Weighted loss for imbalance
    pos_weight = compute_class_weights(
        data["transaction"].y[train_mask].cpu()
    ).to(device)
    criterion = nn.BCELoss()

    history = {
        "train_loss": [], "val_loss": [],
        "val_auc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }
    best_val_auc = 0.0
    best_epoch = 0

    log.info("Starting training for %d epochs...", config.GNN_EPOCHS)
    log.info("Device: %s", device)

    for epoch in range(1, config.GNN_EPOCHS + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        probs = model(data.x_dict, data.edge_index_dict)
        train_probs = probs[train_mask]
        train_labels = data["transaction"].y[train_mask].float()

        # Apply class weighting manually via sample weights
        sample_weights = torch.where(
            train_labels == 1, pos_weight.squeeze(), torch.ones_like(train_labels)
        )
        loss = (nn.BCELoss(reduction="none")(train_probs, train_labels) * sample_weights).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validate
        val_metrics = evaluate(model, data, val_mask, device)
        scheduler.step(val_metrics["auc"])

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "AUC: %.4f | P: %.3f | R: %.3f | F1: %.3f | %.1fs",
            epoch, config.GNN_EPOCHS,
            loss.item(), val_metrics["loss"],
            val_metrics["auc"], val_metrics["precision"],
            val_metrics["recall"], val_metrics["f1"],
            elapsed,
        )

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            save_model(model)
            log.info("  ✓ New best model saved (AUC=%.4f)", best_val_auc)

    log.info(
        "Training complete. Best epoch: %d, Best val AUC: %.4f",
        best_epoch, best_val_auc
    )
    return history


def final_evaluation(
    model: nn.Module,
    data: HeteroData,
    test_mask: torch.Tensor,
    device: torch.device,
) -> None:
    """Run final evaluation on the test set and print detailed report."""
    from graph.gnn_model import load_model
    # Reload best checkpoint
    best_model = load_model(data.cpu())
    best_model = best_model.to(device)
    data = data.to(device)

    test_metrics = evaluate(best_model, data, test_mask, device)

    log.info("=" * 60)
    log.info("FINAL TEST SET RESULTS")
    log.info("  AUC-ROC:   %.4f", test_metrics["auc"])
    log.info("  Precision: %.4f", test_metrics["precision"])
    log.info("  Recall:    %.4f", test_metrics["recall"])
    log.info("  F1:        %.4f", test_metrics["f1"])
    log.info("=" * 60)

    cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
    log.info("Confusion Matrix:\n%s", cm)
    log.info("\n%s", classification_report(
        test_metrics["labels"], test_metrics["preds"],
        target_names=["Legitimate", "Fraud"]
    ))

    # Save test results for dashboard
    import json
    results = {
        "test_auc": test_metrics["auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "confusion_matrix": cm.tolist(),
    }
    results_path = config.MODEL_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Test results saved to %s", results_path)


def main() -> None:
    """Entry point for training."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    log.info("Loading graph data from %s", config.GRAPH_DATA_PATH)
    data, node_maps, scaler, feature_cols = load_graph()
    log.info("Graph loaded successfully")

    train_mask, val_mask, test_mask = create_masks(data)

    model = build_model(data)
    log.info("Model built successfully")

    history = train(model, data, train_mask, val_mask, device)

    plot_training_curves(history, config.TRAINING_CURVES_PATH)

    # Reload data to CPU for evaluation
    data, _, _, _ = load_graph()
    final_evaluation(model, data, test_mask, device)


if __name__ == "__main__":
    main()
