"""
Module: run_transfer_learning.py
Purpose: Step 15 (Novelty 1) — Modern Vision Transfer Learning & Cross-Detector Few-Shot Domain Adaptation.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Experiments Conducted:
- Part A (Pooled Seed 42 Comparison on 4,437 Train / 952 Val / 951 Test):
    1. ResNet-18 (1.0s Single-View)
    2. ResNet-18 (Multi-View RGB: [0.5s, 1.0s, 2.0s] mapped to [R, G, B])
    3. ConvNeXt-Tiny (Multi-View RGB)
    4. ViT-B/16 Vision Transformer (Multi-View RGB)
- Part B (Cross-Detector Few-Shot Domain Adaptation across 5 Seeds [42-46]):
    - Source H1 -> Target L1 (0% zero-shot, 5%, 10%, 20% target fine-tuning vs 100% Native L1)
    - Source L1 -> Target H1 (0% zero-shot, 5%, 10%, 20% target fine-tuning vs 100% Native H1)

Saves all tables, figures, best checkpoint (float16), and packs `step15_results.zip`.
Includes `--smoke-test` mode for fast local verification into a scratch directory.
"""

import os
import sys
import json
import copy
import time
import random
import zipfile
import argparse
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
from src.data.split_utils import create_stratified_glitch_split
from src.models.cnn import compute_class_weights
from src.models.transfer_models import TransferLIGOClassifier, count_trainable_parameters
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary
from src.evaluation.plots import plot_confusion_matrix


SEEDS = [42, 43, 44, 45, 46]
FEW_SHOT_FRACTIONS = [0.05, 0.10, 0.20]


def set_all_seeds(seed=42):
    """Fixes Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preload_multiview_rgb(h5_path, metadata_df, views=("0.5s", "1.0s", "2.0s")):
    """
    Loads 3 duration views per glitch from HDF5 into memory as float16 arrays of shape (3, 140, 170).
    Channel mapping: R=0.5s, G=1.0s, B=2.0s.
    Storing in float16 keeps ~6,340 glitches at ~863 MB in RAM while converting cleanly to float32 per batch.
    """
    n_samples = len(metadata_df)
    spec_dict = {}
    t0 = time.time()
    print(f"Preloading {n_samples} 3-view RGB spectrograms ({views}) from {os.path.basename(h5_path)}...")

    with h5py.File(h5_path, "r") as h5_f:
        for i in range(n_samples):
            row = metadata_df.iloc[i]
            gid = row["gravityspy_id"]
            lbl = row["label"]
            stype = row["sample_type"]

            chans = [load_spectrogram(h5_f, lbl, stype, gid, view=v) for v in views]
            spec_dict[gid] = np.stack(chans, axis=0).astype(np.float16)

            if (i + 1) % 2000 == 0 or (i + 1) == n_samples:
                print(f"  Loaded {i + 1}/{n_samples} multi-view arrays ({time.time() - t0:.1f}s elapsed)...")

    total_mb = sum(v.nbytes for v in spec_dict.values()) / (1024 ** 2)
    print(f"Preload complete in {time.time() - t0:.1f}s ({total_mb:.1f} MB in float16 RAM).")
    return spec_dict


def build_tensor_arrays(sub_df, spec_dict, class_to_idx, in_mode="rgb_3view"):
    """
    Constructs input tensor arrays from preloaded 3-view dictionary without extra HDF5 reads.
    - in_mode='rgb_3view' -> shape (N, 3, 140, 170) float16
    - in_mode='1.0s'      -> slices channel index 1 -> shape (N, 1, 140, 170) float16
    """
    if in_mode == "rgb_3view":
        X = np.stack([spec_dict[gid] for gid in sub_df["gravityspy_id"].values], axis=0)
    elif in_mode == "1.0s":
        X = np.stack([spec_dict[gid][1:2] for gid in sub_df["gravityspy_id"].values], axis=0)
    else:
        raise ValueError(f"Unknown in_mode: {in_mode}")

    y = np.array([class_to_idx[lbl] for lbl in sub_df["label"].values], dtype=np.int64)
    return X, y


def sample_few_shot_stratified(df, fraction, seed, min_per_class=2):
    """
    Robust stratified few-shot sampler for small target-detector fractions (5%, 10%, 20%).
    Guarantees every class present in `df` gets at least `min(min_per_class, len(cls_df))` samples,
    then allocates any remaining budget proportionally so rare classes (Chirp, None_of_the_Above) never fail.
    """
    rng = np.random.RandomState(seed + int(round(fraction * 1000)))
    target_total = max(len(df["label"].unique()), int(round(len(df) * fraction)))

    chosen_indices = []
    class_groups = {}
    for cls_name, grp in df.groupby("label"):
        class_groups[cls_name] = grp.index.tolist()

    # Pass 1: Minimum guaranteed quota per class
    leftover_pool = []
    for cls_name in sorted(class_groups.keys()):
        idxs = list(class_groups[cls_name])
        rng.shuffle(idxs)
        take = min(min_per_class, len(idxs))
        chosen_indices.extend(idxs[:take])
        if len(idxs) > take:
            for idx_rem in idxs[take:]:
                leftover_pool.append((idx_rem, cls_name))

    # Pass 2: Fill remaining budget stratified from remaining pool
    remaining_needed = max(0, target_total - len(chosen_indices))
    if remaining_needed > 0 and len(leftover_pool) > 0:
        if remaining_needed >= len(leftover_pool):
            chosen_indices.extend([idx_val for idx_val, _ in leftover_pool])
        else:
            rem_df = df.loc[[idx_val for idx_val, _ in leftover_pool]]
            # Proportional draw weighted by remaining class counts
            counts = rem_df["label"].value_counts()
            probs = rem_df["label"].map(lambda c: counts[c] / len(rem_df)).values
            probs = probs / probs.sum()
            picked = rng.choice(rem_df.index.values, size=remaining_needed, replace=False, p=probs)
            chosen_indices.extend(picked.tolist())

    return df.loc[chosen_indices].copy().reset_index(drop=True)


def evaluate_transfer_model(model, X_eval, y_eval_idx, class_names, device, batch_size=64):
    """Runs inference on `X_eval` and returns predicted labels, softmax probabilities, and full metrics."""
    model.eval()
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    loader = DataLoader(TensorDataset(torch.from_numpy(X_eval)), batch_size=batch_size, shuffle=False)
    use_amp = (device.type == "cuda")

    all_probs = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device, dtype=torch.float32)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(bx)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)
    pred_idx = np.argmax(probs_arr, axis=1)

    y_true_str = [idx_to_class[int(i)] for i in y_eval_idx]
    y_pred_str = [idx_to_class[int(i)] for i in pred_idx]
    metrics = compute_all_metrics(y_true_str, y_pred_str, class_names=class_names)
    return y_pred_str, probs_arr, metrics


def train_transfer_model(
    model, X_train, y_train, X_val, y_val, class_names,
    seed=42, epochs=18, batch_size=64, backbone_lr=1e-4, head_lr=1e-3,
    weight_decay=1e-2, class_weight_power=0.5, patience=5,
    device=None, verbose=True
):
    """
    Fine-tunes a TransferLIGOClassifier using:
    - Smoothed inverse-frequency CrossEntropyLoss
    - Differential AdamW learning rates + CosineAnnealingLR + grad clipping (1.0)
    - Mixed precision (CUDA AMP) when GPU is available
    - Early stopping strictly on Validation Macro-F1
    """
    set_all_seeds(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    num_classes = len(class_names)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    class_weights = compute_class_weights(y_train, num_classes=num_classes, power=class_weight_power).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=min(batch_size, max(2, len(X_train))),
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=64,
        shuffle=False
    )

    best_val_macro = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = []
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    total_start = time.time()

    if verbose:
        print(f"  Training {model.backbone_name} on {device} (AMP={use_amp}) | Train={len(X_train)}, Val={len(X_val)} | Params={count_trainable_parameters(model):,}")

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx = bx.to(device, dtype=torch.float32)
            by = by.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(bx)
                loss = criterion(logits, by)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += float(loss.item()) * bx.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # Validation evaluation
        model.eval()
        val_loss_sum = 0.0
        val_preds_idx = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, dtype=torch.float32)
                by = by.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(bx)
                    loss = criterion(logits, by)
                val_loss_sum += float(loss.item()) * bx.size(0)
                val_preds_idx.append(torch.argmax(logits, dim=1).cpu().numpy())

        val_loss = val_loss_sum / max(1, len(X_val))
        val_preds_idx = np.concatenate(val_preds_idx, axis=0)
        val_true_str = [idx_to_class[int(i)] for i in y_val]
        val_pred_str = [idx_to_class[int(i)] for i in val_preds_idx]
        val_metrics = compute_all_metrics(val_true_str, val_pred_str, class_names=class_names)
        val_acc = val_metrics["accuracy"]
        val_macro = val_metrics["macro_f1"]
        ep_time = time.time() - ep_start

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_macro_f1": round(val_macro, 4),
            "epoch_time_sec": round(ep_time, 2)
        })

        improved = ""
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            improved = " *"
        else:
            no_improve += 1

        if verbose:
            print(
                f"    Ep {epoch:02d}/{epochs:02d} ({ep_time:4.1f}s) | "
                f"Tr Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
                f"Val Acc: {val_acc*100:5.2f}% Macro-F1: {val_macro:.4f}{improved}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} (patience={patience}).")
            break

    total_time = time.time() - total_start
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), float(best_val_macro), float(total_time)


def plot_transfer_benchmark(transfer_df, baseline_csv_path, save_path):
    """Plots full leaderboard comparing Steps 7-12 models vs Step 15 Pretrained Transfer Backbones."""
    transfer_names = set(transfer_df["Model"].tolist())
    rows = []
    if os.path.exists(baseline_csv_path):
        base_df = pd.read_csv(baseline_csv_path)
        for _, r in base_df.iterrows():
            if r["Model"] in transfer_names:
                continue
            rows.append({
                "Model": r["Model"],
                "Macro_F1": float(r["Macro-F1"]),
                "Accuracy": float(r["Accuracy (%)"]),
                "Group": "Prior"
            })
    for _, r in transfer_df.iterrows():
        rows.append({
            "Model": r["Model"],
            "Macro_F1": float(r["Macro-F1"]),
            "Accuracy": float(r["Accuracy (%)"]),
            "Group": "Transfer"
        })

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12.5, 6))
    colors = ["#7f7f7f" if g == "Prior" else "#1f77b4" for g in plot_df["Group"]]
    bars = ax.barh(plot_df["Model"], plot_df["Macro_F1"], color=colors, edgecolor="black", alpha=0.88)

    for bar, acc in zip(bars, plot_df["Accuracy"]):
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w:.4f} ({acc:.1f}% acc)", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0.0, 1.24)
    ax.set_xlabel("Unseen Test Macro-F1 Score (Primary Metric)", fontsize=11, fontweight="bold")
    ax.set_title("Step 15 Benchmark: Prior Baselines & CNN vs. Modern Pretrained Vision Backbones", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Benchmark figure saved to: {save_path}")


def plot_few_shot_recovery(summary_df, cross_detector_csv, save_path):
    """
    Plots 2-panel Few-Shot Cross-Detector Recovery Curves (H1->L1 and L1->H1) with ±1 Std bands,
    overlaying Step 13 Zero-Shot RF and Adapted CNN baselines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Load Step 13 references if present
    prior_refs = {}
    if os.path.exists(cross_detector_csv):
        cd = pd.read_csv(cross_detector_csv)
        for _, r in cd.iterrows():
            prior_refs[(r["Model"], r["Direction"])] = float(r["Macro_F1_Mean"])

    stage_order = ["0% (Zero-Shot)", "5% Fine-Tuned", "10% Fine-Tuned", "20% Fine-Tuned", "100% Native Target"]
    x_pos = np.arange(len(stage_order))

    for ax, direction, title in [
        (axes[0], "H1->L1", "Hanford (H1) → Livingston (L1) Adaptation"),
        (axes[1], "L1->H1", "Livingston (L1) → Hanford (H1) Adaptation")
    ]:
        sub = summary_df[summary_df["Direction"] == direction].set_index("Stage").loc[stage_order]
        means = sub["Macro_F1_Mean"].values
        stds = sub["Macro_F1_Std"].values

        ax.plot(x_pos, means, marker="o", linewidth=2.5, markersize=8, color="#1f77b4", label="ResNet-18 (Multi-View RGB)")
        ax.fill_between(x_pos, np.clip(means - stds, 0, 1), np.clip(means + stds, 0, 1), color="#1f77b4", alpha=0.18)

        for xi, m, sd in zip(x_pos, means, stds):
            ax.text(xi, m + sd + 0.015, f"{m:.3f}\n±{sd:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Overlay Step 13 baselines
        if ("Feature Random Forest", direction) in prior_refs:
            rf_ref = prior_refs[("Feature Random Forest", direction)]
            ax.axhline(y=rf_ref, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Step 13 Zero-Shot RF ({rf_ref:.3f})")
        if ("Adapted Gravity Spy CNN", direction) in prior_refs:
            cnn_ref = prior_refs[("Adapted Gravity Spy CNN", direction)]
            ax.axhline(y=cnn_ref, color="#ff7f0e", linestyle=":", linewidth=1.8, label=f"Step 13 Zero-Shot 4-Layer CNN ({cnn_ref:.3f})")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["0%\n(Zero-Shot)", "5%\nTarget Data", "10%\nTarget Data", "20%\nTarget Data", "100%\nNative Upper"], fontsize=9.5, fontweight="bold")
        ax.set_ylim(0.55, 1.05)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Labeled Target-Detector Data Used for Fine-Tuning", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    axes[0].set_ylabel("Target-Site Test Macro-F1 (Mean ± Std)", fontsize=11, fontweight="bold")
    plt.suptitle("Step 15 Novelty: Cross-Detector Few-Shot Domain Adaptation Recovery Curves", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Few-shot recovery curve saved to: {save_path}")


def package_step15_results_zip(output_dir, zip_path, generated_rel_paths):
    """Packages only the generated Step 15 artifacts cleanly into `step15_results.zip`."""
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_subpath in generated_rel_paths:
            src_abs = os.path.join(output_dir, rel_subpath)
            if os.path.exists(src_abs):
                arcname = os.path.join("outputs", rel_subpath).replace("\\", "/")
                zf.write(src_abs, arcname=arcname)
    size_mb = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"\nPackaged Step 15 deliverables into: {zip_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Step 15: Modern Vision Transfer Learning & Cross-Detector Few-Shot Adaptation")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast local verification on tiny subset with pretrained=False, 1 epoch, seed 42.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "outputs"), help="Directory for tables/, figures/, models/.")
    parser.add_argument("--h5-path", type=str, default=os.path.join(project_root, "trainingsetv1d1.h5"), help="Path to trainingsetv1d1.h5.")
    parser.add_argument("--zip-path", type=str, default=os.path.join(project_root, "step15_results.zip"), help="Destination path for results zip archive.")
    args = parser.parse_args()

    meta_csv = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    prior_bench_csv = os.path.join(project_root, "outputs", "tables", "benchmark_comparison.csv")
    prior_cd_csv = os.path.join(project_root, "outputs", "tables", "cross_detector_summary.csv")

    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(" STEP 15: MODERN VISION TRANSFER LEARNING & CROSS-DETECTOR DOMAIN ADAPTATION")
    print("=" * 80)
    print(f"Mode       : {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")
    print(f"Device     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Output Dir : {args.output_dir}")

    meta_df = pd.read_csv(meta_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, _ = get_class_mapping(class_names)
    default_split = split_registry["default_split"]

    if args.smoke_test:
        # Take up to 10 samples per class per detector so all 14 classes stay intact in H1 & L1
        smoke_parts = []
        for (cls_name, ifo_name), grp in meta_df.groupby(["label", "ifo"]):
            smoke_parts.append(grp.head(10))
        work_meta = pd.concat(smoke_parts, ignore_index=True)

        tr_df, va_df, te_df = [], [], []
        split_s = create_stratified_glitch_split(work_meta, test_size=0.20, val_size=0.15, random_state=42)
        tr_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["train_ids"]))].reset_index(drop=True)
        va_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["val_ids"]))].reset_index(drop=True)
        te_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["test_ids"]))].reset_index(drop=True)

        use_pretrained = False
        pooled_epochs = 1
        pooled_patience = 1
        part_b_seeds = [42]
        part_b_fracs = [0.10]
        src_epochs = 1
        ft_epochs = 1
    else:
        work_meta = meta_df
        tr_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["train_ids"]))].copy().reset_index(drop=True)
        va_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["val_ids"]))].copy().reset_index(drop=True)
        te_df = meta_df[meta_df["gravityspy_id"].isin(set(default_split["test_ids"]))].copy().reset_index(drop=True)

        use_pretrained = True
        pooled_epochs = 18
        pooled_patience = 5
        part_b_seeds = SEEDS
        part_b_fracs = FEW_SHOT_FRACTIONS
        src_epochs = 12
        ft_epochs = 10

    # Preload 3-view RGB spectrograms once
    spec_dict = preload_multiview_rgb(args.h5_path, work_meta, views=("0.5s", "1.0s", "2.0s"))

    # =========================================================================
    # PART A: POOLED SEED 42 TRANSFER LEARNING BENCHMARK (4 CONFIGURATIONS)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" PART A: POOLED TRANSFER LEARNING BACKBONE COMPARISON (SEED 42)")
    print("=" * 80)

    pooled_configs = [
        {"label": "ResNet-18 (1.0s Single-View)",   "backbone": "resnet18",      "in_mode": "1.0s",      "b_lr": 2e-4, "h_lr": 1e-3, "bs": 64, "ep": pooled_epochs},
        {"label": "ResNet-18 (Multi-View RGB)",     "backbone": "resnet18",      "in_mode": "rgb_3view", "b_lr": 2e-4, "h_lr": 1e-3, "bs": 64, "ep": pooled_epochs},
        {"label": "ConvNeXt-Tiny (Multi-View RGB)", "backbone": "convnext_tiny", "in_mode": "rgb_3view", "b_lr": 1e-4, "h_lr": 1e-3, "bs": 32, "ep": min(pooled_epochs, 16)},
        {"label": "ViT-B/16 (Multi-View RGB)",      "backbone": "vit_b_16",      "in_mode": "rgb_3view", "b_lr": 5e-5, "h_lr": 5e-4, "bs": 32, "ep": min(pooled_epochs, 16)},
    ]

    pooled_results = []
    all_histories = []
    per_class_df = pd.DataFrame({"Class": class_names})
    best_overall_macro = -1.0
    best_overall_info = None
    best_overall_preds = None
    best_overall_probs = None
    best_overall_cm = None

    for cfg in pooled_configs:
        print(f"\n--- Training [{cfg['label']}] ---")
        X_tr, y_tr = build_tensor_arrays(tr_df, spec_dict, class_to_idx, in_mode=cfg["in_mode"])
        X_va, y_va = build_tensor_arrays(va_df, spec_dict, class_to_idx, in_mode=cfg["in_mode"])
        X_te, y_te = build_tensor_arrays(te_df, spec_dict, class_to_idx, in_mode=cfg["in_mode"])

        net = TransferLIGOClassifier(backbone_name=cfg["backbone"], num_classes=len(class_names), pretrained=use_pretrained, dropout_rate=0.3)
        trained_net, hist_df, best_val_mf1, t_sec = train_transfer_model(
            net, X_tr, y_tr, X_va, y_va, class_names,
            seed=42, epochs=cfg["ep"], batch_size=cfg["bs"],
            backbone_lr=cfg["b_lr"], head_lr=cfg["h_lr"],
            patience=pooled_patience, device=device, verbose=True
        )
        hist_df["model"] = cfg["label"]
        all_histories.append(hist_df)

        y_pred, probs, metrics = evaluate_transfer_model(trained_net, X_te, y_te, class_names, device, batch_size=64)
        print_metrics_summary(cfg["label"], metrics)

        pooled_results.append({
            "Model": cfg["label"],
            "Backbone": cfg["backbone"],
            "Input_Mode": cfg["in_mode"],
            "Params": count_trainable_parameters(trained_net),
            "Val_Macro_F1": round(best_val_mf1, 4),
            "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
            "Macro-F1": round(metrics["macro_f1"], 4),
            "Weighted-F1": round(metrics["weighted_f1"], 4),
            "Train_Time_Sec": round(t_sec, 1)
        })

        f1_map = dict(zip(metrics["per_class"]["Class"], metrics["per_class"]["F1-Score"]))
        per_class_df[cfg["label"]] = [f1_map[c] for c in class_names]

        if metrics["macro_f1"] > best_overall_macro:
            best_overall_macro = metrics["macro_f1"]
            best_overall_preds = y_pred
            best_overall_probs = probs
            best_overall_cm = metrics["confusion_matrix"]
            # Store float16 weights on CPU to keep step15_results.zip compact
            half_state = {k: (v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k, v in trained_net.state_dict().items()}
            best_overall_info = {
                "model_label": cfg["label"],
                "backbone_name": cfg["backbone"],
                "in_mode": cfg["in_mode"],
                "class_names": class_names,
                "state_dict_fp16": half_state,
                "test_macro_f1": metrics["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "val_macro_f1": best_val_mf1,
            }

        del trained_net, net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_df = pd.DataFrame(pooled_results)
    pooled_df.to_csv(os.path.join(tables_dir, "transfer_learning_pooled_results.csv"), index=False)
    per_class_df.to_csv(os.path.join(tables_dir, "transfer_learning_per_class_f1.csv"), index=False)
    pd.concat(all_histories, ignore_index=True).to_csv(os.path.join(tables_dir, "transfer_learning_histories.csv"), index=False)

    # Save best model checkpoint & test predictions
    torch.save(best_overall_info, os.path.join(models_dir, "transfer_best_checkpoint.pt"))
    pd.DataFrame({
        "gravityspy_id": te_df["gravityspy_id"].values,
        "ifo": te_df["ifo"].values,
        "sample_type": te_df["sample_type"].values,
        "true_label": te_df["label"].values,
        "transfer_best_pred": best_overall_preds,
        "transfer_best_model": best_overall_info["model_label"],
        "confidence": np.round(np.max(best_overall_probs, axis=1), 4),
        "correct": (te_df["label"].values == np.array(best_overall_preds))
    }).to_csv(os.path.join(tables_dir, "transfer_best_test_predictions.csv"), index=False)

    plot_transfer_benchmark(pooled_df, prior_bench_csv, os.path.join(figures_dir, "transfer_learning_benchmark.png"))
    plot_confusion_matrix(
        best_overall_cm, class_names,
        title=f"Normalized Confusion Matrix: {best_overall_info['model_label']}\nUnseen Test Set (Macro-F1={best_overall_macro:.4f})",
        save_path=os.path.join(figures_dir, "confusion_matrix_transfer_best.png")
    )

    print("\n=== PART A COMPLETE LEADERBOARD ===")
    print(pooled_df.to_string(index=False))

    # =========================================================================
    # PART B: CROSS-DETECTOR FEW-SHOT DOMAIN ADAPTATION (H1 <-> L1, 5 SEEDS)
    # =========================================================================
    print("\n" + "=" * 80)
    print(f" PART B: CROSS-DETECTOR FEW-SHOT DOMAIN ADAPTATION (SEEDS={part_b_seeds})")
    print("=" * 80)

    h1_meta = work_meta[work_meta["ifo"] == "H1"].copy().reset_index(drop=True)
    l1_meta = work_meta[work_meta["ifo"] == "L1"].copy().reset_index(drop=True)

    few_shot_records = []
    stage_names = ["0% (Zero-Shot)", "5% Fine-Tuned", "10% Fine-Tuned", "20% Fine-Tuned", "100% Native Target"]
    if args.smoke_test:
        stage_names = ["0% (Zero-Shot)", "10% Fine-Tuned", "100% Native Target"]

    pc_accum = {
        (d, st): {c: [] for c in class_names}
        for d in ["H1->L1", "L1->H1"]
        for st in stage_names
    }

    for seed in part_b_seeds:
        print(f"\n--- Cross-Detector Adaptation Seed {seed} ---")
        h1_sp = create_stratified_glitch_split(h1_meta, test_size=0.15, val_size=0.15, random_state=seed)
        l1_sp = create_stratified_glitch_split(l1_meta, test_size=0.15, val_size=0.15, random_state=seed)

        h1_tr = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_sp["train_ids"]))].reset_index(drop=True)
        h1_va = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_sp["val_ids"]))].reset_index(drop=True)
        h1_te = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_sp["test_ids"]))].reset_index(drop=True)

        l1_tr = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_sp["train_ids"]))].reset_index(drop=True)
        l1_va = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_sp["val_ids"]))].reset_index(drop=True)
        l1_te = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_sp["test_ids"]))].reset_index(drop=True)

        assert set(h1_tr["gravityspy_id"]).isdisjoint(set(h1_te["gravityspy_id"]))
        assert set(l1_tr["gravityspy_id"]).isdisjoint(set(l1_te["gravityspy_id"]))

        X_h1_tr, y_h1_tr = build_tensor_arrays(h1_tr, spec_dict, class_to_idx, "rgb_3view")
        X_h1_va, y_h1_va = build_tensor_arrays(h1_va, spec_dict, class_to_idx, "rgb_3view")
        X_h1_te, y_h1_te = build_tensor_arrays(h1_te, spec_dict, class_to_idx, "rgb_3view")

        X_l1_tr, y_l1_tr = build_tensor_arrays(l1_tr, spec_dict, class_to_idx, "rgb_3view")
        X_l1_va, y_l1_va = build_tensor_arrays(l1_va, spec_dict, class_to_idx, "rgb_3view")
        X_l1_te, y_l1_te = build_tensor_arrays(l1_te, spec_dict, class_to_idx, "rgb_3view")

        # 1. Train Native Source Models (100% H1 and 100% L1)
        src_h1_net, _, _, _ = train_transfer_model(
            TransferLIGOClassifier("resnet18", len(class_names), pretrained=use_pretrained, dropout_rate=0.3),
            X_h1_tr, y_h1_tr, X_h1_va, y_h1_va, class_names,
            seed=seed, epochs=src_epochs, batch_size=64, backbone_lr=2e-4, head_lr=1e-3,
            patience=4, device=device, verbose=False
        )
        src_l1_net, _, _, _ = train_transfer_model(
            TransferLIGOClassifier("resnet18", len(class_names), pretrained=use_pretrained, dropout_rate=0.3),
            X_l1_tr, y_l1_tr, X_l1_va, y_l1_va, class_names,
            seed=seed, epochs=src_epochs, batch_size=64, backbone_lr=2e-4, head_lr=1e-3,
            patience=4, device=device, verbose=False
        )

        # Save source CPU state dicts cleanly
        src_h1_state = copy.deepcopy(src_h1_net.state_dict())
        src_l1_state = copy.deepcopy(src_l1_net.state_dict())

        # Record 100% Native Targets (upper bounds for H1->L1 is Native L1; for L1->H1 is Native H1)
        _, _, m_native_l1 = evaluate_transfer_model(src_l1_net, X_l1_te, y_l1_te, class_names, device)
        _, _, m_native_h1 = evaluate_transfer_model(src_h1_net, X_h1_te, y_h1_te, class_names, device)

        # Record 0% Zero-Shot Cross-Detector
        _, _, m_zero_h1_to_l1 = evaluate_transfer_model(src_h1_net, X_l1_te, y_l1_te, class_names, device)
        _, _, m_zero_l1_to_h1 = evaluate_transfer_model(src_l1_net, X_h1_te, y_h1_te, class_names, device)

        for direction, stage_lbl, m_res in [
            ("H1->L1", "0% (Zero-Shot)", m_zero_h1_to_l1),
            ("H1->L1", "100% Native Target", m_native_l1),
            ("L1->H1", "0% (Zero-Shot)", m_zero_l1_to_h1),
            ("L1->H1", "100% Native Target", m_native_h1),
        ]:
            few_shot_records.append({
                "Direction": direction, "Stage": stage_lbl, "Seed": seed,
                "Target_Train_Samples": 0 if stage_lbl.startswith("0%") else (len(l1_tr) if direction == "H1->L1" else len(h1_tr)),
                "Accuracy (%)": round(m_res["accuracy"] * 100, 2),
                "Macro_F1": round(m_res["macro_f1"], 4),
                "Weighted_F1": round(m_res["weighted_f1"], 4),
            })
            for cls_name, f1_v in zip(m_res["per_class"]["Class"], m_res["per_class"]["F1-Score"]):
                pc_accum[(direction, stage_lbl)][cls_name].append(float(f1_v))

        print(f"  [Seed {seed}] H1->L1 Zero-Shot={m_zero_h1_to_l1['macro_f1']:.4f} (Native L1={m_native_l1['macro_f1']:.4f}) | "
              f"L1->H1 Zero-Shot={m_zero_l1_to_h1['macro_f1']:.4f} (Native H1={m_native_h1['macro_f1']:.4f})")

        # 2. Few-Shot Target Adaptation at 5%, 10%, 20%
        for frac in part_b_fracs:
            stage_lbl = f"{int(round(frac * 100))}% Fine-Tuned"

            # H1 -> L1: Start from H1 weights, fine-tune on few-shot L1 sample, validate on L1 val, test on L1 test
            sub_l1 = sample_few_shot_stratified(l1_tr, fraction=frac, seed=seed, min_per_class=2)
            X_sub_l1, y_sub_l1 = build_tensor_arrays(sub_l1, spec_dict, class_to_idx, "rgb_3view")
            ft_h1_to_l1 = TransferLIGOClassifier("resnet18", len(class_names), pretrained=False, dropout_rate=0.3)
            ft_h1_to_l1.load_state_dict(copy.deepcopy(src_h1_state))
            ft_h1_to_l1, _, _, _ = train_transfer_model(
                ft_h1_to_l1, X_sub_l1, y_sub_l1, X_l1_va, y_l1_va, class_names,
                seed=seed, epochs=ft_epochs, batch_size=32, backbone_lr=1e-4, head_lr=5e-4,
                patience=4, device=device, verbose=False
            )
            _, _, m_ft_h1_l1 = evaluate_transfer_model(ft_h1_to_l1, X_l1_te, y_l1_te, class_names, device)

            # L1 -> H1: Start from L1 weights, fine-tune on few-shot H1 sample, validate on H1 val, test on H1 test
            sub_h1 = sample_few_shot_stratified(h1_tr, fraction=frac, seed=seed, min_per_class=2)
            X_sub_h1, y_sub_h1 = build_tensor_arrays(sub_h1, spec_dict, class_to_idx, "rgb_3view")
            ft_l1_to_h1 = TransferLIGOClassifier("resnet18", len(class_names), pretrained=False, dropout_rate=0.3)
            ft_l1_to_h1.load_state_dict(copy.deepcopy(src_l1_state))
            ft_l1_to_h1, _, _, _ = train_transfer_model(
                ft_l1_to_h1, X_sub_h1, y_sub_h1, X_h1_va, y_h1_va, class_names,
                seed=seed, epochs=ft_epochs, batch_size=32, backbone_lr=1e-4, head_lr=5e-4,
                patience=4, device=device, verbose=False
            )
            _, _, m_ft_l1_h1 = evaluate_transfer_model(ft_l1_to_h1, X_h1_te, y_h1_te, class_names, device)

            for direction, sub_n, m_res in [
                ("H1->L1", len(sub_l1), m_ft_h1_l1),
                ("L1->H1", len(sub_h1), m_ft_l1_h1),
            ]:
                few_shot_records.append({
                    "Direction": direction, "Stage": stage_lbl, "Seed": seed,
                    "Target_Train_Samples": sub_n,
                    "Accuracy (%)": round(m_res["accuracy"] * 100, 2),
                    "Macro_F1": round(m_res["macro_f1"], 4),
                    "Weighted_F1": round(m_res["weighted_f1"], 4),
                })
                for cls_name, f1_v in zip(m_res["per_class"]["Class"], m_res["per_class"]["F1-Score"]):
                    pc_accum[(direction, stage_lbl)][cls_name].append(float(f1_v))

            print(f"    [{stage_lbl:14s}] H1->L1 ({len(sub_l1):3d} L1 samp): {m_ft_h1_l1['macro_f1']:.4f} | "
                  f"L1->H1 ({len(sub_h1):3d} H1 samp): {m_ft_l1_h1['macro_f1']:.4f}")

            del ft_h1_to_l1, ft_l1_to_h1

        del src_h1_net, src_l1_net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate Part B tables
    fs_seed_df = pd.DataFrame(few_shot_records)
    fs_seed_df.to_csv(os.path.join(tables_dir, "few_shot_adaptation_per_seed.csv"), index=False)

    summary_rows = []
    for direction in ["H1->L1", "L1->H1"]:
        sub_d = fs_seed_df[fs_seed_df["Direction"] == direction]
        zero_mean = float(sub_d[sub_d["Stage"] == "0% (Zero-Shot)"]["Macro_F1"].mean())
        for st in stage_names:
            sub_s = sub_d[sub_d["Stage"] == st]
            mf1_m = float(sub_s["Macro_F1"].mean())
            mf1_sd = float(sub_s["Macro_F1"].std(ddof=0))
            acc_m = float(sub_s["Accuracy (%)"].mean())
            acc_sd = float(sub_s["Accuracy (%)"].std(ddof=0))
            n_samp = int(round(sub_s["Target_Train_Samples"].mean()))
            summary_rows.append({
                "Direction": direction,
                "Stage": st,
                "Avg_Target_Train_Samples": n_samp,
                "Accuracy_Mean (%)": round(acc_m, 2),
                "Accuracy_Std (%)": round(acc_sd, 2),
                "Macro_F1_Mean": round(mf1_m, 4),
                "Macro_F1_Std": round(mf1_sd, 4),
                "Gain_vs_ZeroShot": round(mf1_m - zero_mean, 4),
                "Num_Seeds": len(part_b_seeds)
            })

    fs_summary_df = pd.DataFrame(summary_rows)
    fs_summary_df.to_csv(os.path.join(tables_dir, "few_shot_adaptation_summary.csv"), index=False)

    pc_rec_rows = []
    for direction in ["H1->L1", "L1->H1"]:
        for cls_name in class_names:
            r = {"Direction": direction, "Class": cls_name}
            for st in stage_names:
                vals = pc_accum[(direction, st)][cls_name]
                r[st] = round(float(np.mean(vals)), 4)
            pc_rec_rows.append(r)
    pd.DataFrame(pc_rec_rows).to_csv(os.path.join(tables_dir, "few_shot_per_class_recovery.csv"), index=False)

    print("\n=== PART B SUMMARY: FEW-SHOT CROSS-DETECTOR ADAPTATION ===")
    print(fs_summary_df.to_string(index=False))

    if not args.smoke_test:
        plot_few_shot_recovery(fs_summary_df, prior_cd_csv, os.path.join(figures_dir, "few_shot_cross_detector_curve.png"))
    else:
        # In smoke test, plot using available stages cleanly
        fig_sm, ax_sm = plt.subplots(figsize=(6, 4))
        for d in ["H1->L1", "L1->H1"]:
            sd = fs_summary_df[fs_summary_df["Direction"] == d]
            ax_sm.plot(sd["Stage"], sd["Macro_F1_Mean"], marker="o", label=d)
        ax_sm.set_title("Smoke Few-Shot Plot")
        ax_sm.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "few_shot_cross_detector_curve.png"), dpi=120)
        plt.close(fig_sm)

    # Package all generated Step 15 files into zip
    step15_artifacts = [
        "tables/transfer_learning_pooled_results.csv",
        "tables/transfer_learning_per_class_f1.csv",
        "tables/transfer_learning_histories.csv",
        "tables/transfer_best_test_predictions.csv",
        "tables/few_shot_adaptation_per_seed.csv",
        "tables/few_shot_adaptation_summary.csv",
        "tables/few_shot_per_class_recovery.csv",
        "figures/transfer_learning_benchmark.png",
        "figures/confusion_matrix_transfer_best.png",
        "figures/few_shot_cross_detector_curve.png",
        "models/transfer_best_checkpoint.pt",
    ]
    package_step15_results_zip(args.output_dir, args.zip_path, step15_artifacts)
    print("\nStep 15 execution finished successfully!")


if __name__ == "__main__":
    main()
