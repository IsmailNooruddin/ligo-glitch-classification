"""
Module: run_cnn.py
Purpose: Trains and evaluates the Adapted Gravity Spy CNN (Step 12) on the 14 shared classes.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Key Design Choices:
1. Preloads spectrogram arrays from trainingsetv1d1.h5 into memory once to eliminate disk I/O bottlenecks.
2. Uses smoothed inverse-frequency class weights (`power=0.5`) inside CrossEntropyLoss to handle class imbalance.
3. Optimizes with AdamW + ReduceLROnPlateau and early stopping strictly monitoring Validation Macro-F1 (zero test peeking).
4. Saves best checkpoint (`outputs/models/cnn_best_checkpoint.pt`), training history, per-sample test predictions,
   learning curves, confusion matrix heatmap, and updates global benchmark comparison tables.
"""

import os
import sys
import json
import copy
import time
import random
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_spectrogram, get_class_mapping
from src.models.cnn import AdaptedGravitySpyCNN, compute_class_weights, count_parameters
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary
from src.evaluation.plots import plot_confusion_matrix


def set_all_seeds(seed=42):
    """Sets Python, NumPy, and PyTorch random seeds for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preload_spectrogram_dict(h5_path, metadata_df, view="1.0s"):
    """
    Loads all spectrograms for rows in metadata_df from HDF5 into an in-memory dictionary
    mapping `gravityspy_id` -> float32 ndarray of shape (C, 140, 170).
    """
    n_samples = len(metadata_df)
    spec_dict = {}
    t0 = time.time()
    print(f"Preloading {n_samples} spectrograms from HDF5 (view='{view}')...")

    with h5py.File(h5_path, "r") as h5_f:
        for i in range(n_samples):
            row = metadata_df.iloc[i]
            gid = row["gravityspy_id"]
            lbl = row["label"]
            stype = row["sample_type"]

            if view == "all":
                chans = [load_spectrogram(h5_f, lbl, stype, gid, view=v) for v in ["0.5s", "1.0s", "2.0s", "4.0s"]]
                arr = np.stack(chans, axis=0)  # (4, 140, 170)
            else:
                arr = np.expand_dims(load_spectrogram(h5_f, lbl, stype, gid, view=view), axis=0)  # (1, 140, 170)

            spec_dict[gid] = arr

            if (i + 1) % 2000 == 0 or (i + 1) == n_samples:
                print(f"  Loaded {i + 1}/{n_samples} arrays ({time.time() - t0:.1f}s elapsed)...")

    total_mb = sum(v.nbytes for v in spec_dict.values()) / (1024 ** 2)
    print(f"Preload complete in {time.time() - t0:.1f}s ({total_mb:.1f} MB in RAM).")
    return spec_dict


def build_tensor_arrays(sub_df, spec_dict, class_to_idx):
    """Stacks preloaded spectrogram arrays for a subset DataFrame into (X, y) numpy arrays."""
    X = np.stack([spec_dict[gid] for gid in sub_df["gravityspy_id"].values], axis=0).astype(np.float32)
    y = np.array([class_to_idx[lbl] for lbl in sub_df["label"].values], dtype=np.int64)
    return X, y


def evaluate_cnn_model(model, X_eval, y_eval_idx, class_names, device, batch_size=128):
    """
    Runs inference in eval mode and returns predicted string labels, softmax probabilities,
    and full classification metrics dictionary.
    """
    model.eval()
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    loader = DataLoader(TensorDataset(torch.from_numpy(X_eval)), batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)
    pred_idx = np.argmax(probs_arr, axis=1)

    y_true_str = [idx_to_class[int(i)] for i in y_eval_idx]
    y_pred_str = [idx_to_class[int(i)] for i in pred_idx]
    metrics = compute_all_metrics(y_true_str, y_pred_str, class_names=class_names)
    return y_pred_str, probs_arr, metrics


def train_cnn_model(
    X_train, y_train, X_val, y_val, class_names,
    in_channels=1, seed=42, epochs=25, batch_size=64,
    lr=1e-3, weight_decay=1e-4, dropout_rate=0.4,
    class_weight_power=0.5, patience=6, device=None, verbose=True
):
    """
    Trains AdaptedGravitySpyCNN with class-weighted CrossEntropyLoss, AdamW, LR scheduling,
    and early stopping based on Validation Macro-F1.

    Returns:
    --------
    best_model : AdaptedGravitySpyCNN (loaded with best validation weights)
    history_df : pd.DataFrame (per-epoch train/val loss, accuracy, macro-F1)
    best_val_macro_f1 : float
    total_train_time : float
    """
    set_all_seeds(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = len(class_names)
    model = AdaptedGravitySpyCNN(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate
    ).to(device)

    # Smoothed inverse-frequency class weights for imbalanced training set
    class_weights = compute_class_weights(y_train, num_classes=num_classes, power=class_weight_power).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=128,
        shuffle=False
    )

    best_val_macro = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve_epochs = 0
    history = []
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    total_start = time.time()

    if verbose:
        print(f"Training AdaptedGravitySpyCNN on {device} | Train={len(X_train)}, Val={len(X_val)} | Params={count_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_preds_idx = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_sum += loss.item() * bx.size(0)
                val_preds_idx.append(torch.argmax(logits, dim=1).cpu().numpy())

        val_loss = val_loss_sum / len(X_val)
        val_preds_idx = np.concatenate(val_preds_idx, axis=0)
        val_true_str = [idx_to_class[int(i)] for i in y_val]
        val_pred_str = [idx_to_class[int(i)] for i in val_preds_idx]
        val_metrics = compute_all_metrics(val_true_str, val_pred_str, class_names=class_names)
        val_acc = val_metrics["accuracy"]
        val_macro = val_metrics["macro_f1"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_macro)
        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_macro_f1": round(val_macro, 4),
            "lr": current_lr,
            "epoch_time_sec": round(ep_time, 2)
        })

        improved = ""
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            improved = " *"
        else:
            no_improve_epochs += 1

        if verbose:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} ({ep_time:4.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% Macro-F1: {val_macro:.4f}{improved}"
            )

        if no_improve_epochs >= patience:
            if verbose:
                print(f"  Early stopping triggered at epoch {epoch} (no Val Macro-F1 improvement for {patience} epochs).")
            break

    total_time = time.time() - total_start
    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    return model, history_df, float(best_val_macro), float(total_time)


def plot_training_curves(history_df, save_path):
    """Plots Training/Validation Loss and Validation Macro-F1/Accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Loss curves
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker='o', label="Train Loss", color="#1f77b4", linewidth=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker='s', label="Validation Loss", color="#d62728", linewidth=2)
    axes[0].set_title("Adapted Gravity Spy CNN: Cross-Entropy Loss", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10, fontweight='bold')
    axes[0].set_ylabel("Class-Weighted Loss", fontsize=10, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # Right: Accuracy & Macro-F1
    axes[1].plot(history_df["epoch"], history_df["val_macro_f1"] * 100, marker='o', label="Val Macro-F1 (%)", color="#2ca02c", linewidth=2.2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"], marker='^', label="Val Accuracy (%)", color="#ff7f0e", linewidth=1.8, linestyle='--')
    axes[1].plot(history_df["epoch"], history_df["train_acc"], marker='.', label="Train Accuracy (%)", color="#1f77b4", linewidth=1.4, alpha=0.7)
    best_idx = history_df["val_macro_f1"].idxmax()
    best_ep = history_df.loc[best_idx, "epoch"]
    best_f1 = history_df.loc[best_idx, "val_macro_f1"] * 100
    axes[1].axvline(x=best_ep, color="gray", linestyle=":", label=f"Best Checkpoint (Ep {best_ep}: {best_f1:.2f}%)")
    axes[1].set_title("Adapted Gravity Spy CNN: Validation Performance", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=10, fontweight='bold')
    axes[1].set_ylabel("Score (%)", fontsize=10, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=9)

    plt.suptitle("Step 12: Adapted Gravity Spy CNN Training Trajectory (Seed 42, 1.0s View)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Training curves saved to: {save_path}")


def main():
    meta_csv = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    h5_path = os.path.join(project_root, "trainingsetv1d1.h5")
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")
    models_dir = os.path.join(project_root, "outputs", "models")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 72)
    print(" STEP 12: ADAPTED GRAVITY SPY CNN TRAINING & BENCHMARK EVALUATION")
    print("=" * 72)

    # 1. Load metadata and leak-free Seed 42 split
    meta_df = pd.read_csv(meta_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, idx_to_class = get_class_mapping(class_names)
    default_split = split_registry["default_split"]

    train_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["train_ids"]))].copy().reset_index(drop=True)
    val_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["val_ids"]))].copy().reset_index(drop=True)
    test_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["test_ids"]))].copy().reset_index(drop=True)

    print(f"Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} | Classes={len(class_names)}")

    # 2. Preload all 6,340 1.0s spectrograms into RAM
    spec_dict = preload_spectrogram_dict(h5_path, meta_df, view="1.0s")

    X_train, y_train = build_tensor_arrays(train_df, spec_dict, class_to_idx)
    X_val, y_val = build_tensor_arrays(val_df, spec_dict, class_to_idx)
    X_test, y_test = build_tensor_arrays(test_df, spec_dict, class_to_idx)

    # 3. Train Adapted Gravity Spy CNN (Single-View 1.0s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model, history_df, best_val_macro, train_time = train_cnn_model(
        X_train, y_train, X_val, y_val, class_names,
        in_channels=1, seed=42, epochs=25, batch_size=64,
        lr=1e-3, weight_decay=1e-4, dropout_rate=0.4,
        class_weight_power=0.5, patience=6, device=device, verbose=True
    )

    print(f"\nCNN Training Complete in {train_time:.1f}s | Best Val Macro-F1: {best_val_macro:.4f}")

    # 4. Evaluate best checkpoint on unseen Test set (N=951)
    y_test_pred, test_probs, test_metrics = evaluate_cnn_model(best_model, X_test, y_test, class_names, device)
    print_metrics_summary("Adapted Gravity Spy CNN (1.0s View, Seed 42)", test_metrics)

    # 5. Save checkpoint & delete placeholder
    ckpt_path = os.path.join(models_dir, "cnn_best_checkpoint.pt")
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "in_channels": 1,
        "view": "1.0s",
        "best_val_macro_f1": best_val_macro,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "train_time_sec": train_time,
        "seed": 42
    }, ckpt_path)
    print(f"Saved CNN checkpoint to: {ckpt_path}")

    placeholder_cnn = os.path.join(models_dir, "PLACEHOLDER_cnn_best_checkpoint_LATER.pt.txt")
    if os.path.exists(placeholder_cnn):
        os.remove(placeholder_cnn)
        print(f"Removed placeholder: {placeholder_cnn}")

    # 6. Save training history & per-sample test predictions (for Step 14 failure analysis)
    hist_csv = os.path.join(tables_dir, "cnn_training_history.csv")
    history_df.to_csv(hist_csv, index=False)

    pred_df = pd.DataFrame({
        "gravityspy_id": test_df["gravityspy_id"].values,
        "ifo": test_df["ifo"].values,
        "sample_type": test_df["sample_type"].values,
        "true_label": test_df["label"].values,
        "cnn_pred": y_test_pred,
        "cnn_confidence": np.max(test_probs, axis=1).round(4),
        "cnn_correct": (test_df["label"].values == np.array(y_test_pred))
    })
    pred_csv = os.path.join(tables_dir, "cnn_test_predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved per-sample CNN predictions to: {pred_csv}")

    # 7. Save figures (training curves & confusion matrix)
    plot_training_curves(history_df, os.path.join(figures_dir, "cnn_training_curves.png"))
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        title="Normalized Confusion Matrix: Adapted Gravity Spy CNN (1.0s View)\nEvaluated on Unseen Test Set (N=951)",
        save_path=os.path.join(figures_dir, "confusion_matrix_cnn.png")
    )

    # 8. Update benchmark_comparison.csv
    bench_file = os.path.join(tables_dir, "benchmark_comparison.csv")
    bench_df = pd.read_csv(bench_file) if os.path.exists(bench_file) else pd.DataFrame()
    cnn_entry = {
        "Model": "Adapted Gravity Spy CNN (1.0s)",
        "Accuracy (%)": round(test_metrics["accuracy"] * 100, 2),
        "Macro-F1": round(test_metrics["macro_f1"], 4),
        "Weighted-F1": round(test_metrics["weighted_f1"], 4),
        "Description": "4-block PyTorch CNN (adapted Glanzer et al. 2023) on 1.0s spectrograms with smoothed class weights"
    }
    if not bench_df.empty and cnn_entry["Model"] in bench_df["Model"].values:
        bench_df = bench_df[bench_df["Model"] != cnn_entry["Model"]]
    bench_df = pd.concat([bench_df, pd.DataFrame([cnn_entry])], ignore_index=True)
    bench_df.to_csv(bench_file, index=False)
    print(f"\nUpdated {bench_file}:")
    print(bench_df.to_string(index=False))

    # 9. Update per_class_f1_summary.csv
    pc_file = os.path.join(tables_dir, "per_class_f1_summary.csv")
    pc_df = pd.read_csv(pc_file)
    # Align strictly by Class name
    cnn_f1_map = dict(zip(test_metrics["per_class"]["Class"], test_metrics["per_class"]["F1-Score"]))
    pc_df["CNN_1s_F1"] = [cnn_f1_map[c] for c in pc_df["Class"]]
    pc_df.to_csv(pc_file, index=False)
    print(f"\nUpdated {pc_file} with CNN_1s_F1 column.")
    print(pc_df.to_string(index=False))

    print("\nStep 12 completed successfully!")


if __name__ == "__main__":
    main()
