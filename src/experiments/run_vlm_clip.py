"""
Module: run_vlm_clip.py
Purpose: Step 16 (Novelty 2) — Vision-Language Model (OpenAI CLIP ViT-B/16) Evaluation on LIGO Glitches.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Experiments Conducted:
- Part A (True Zero-Shot Classification, 0 LIGO training labels, Unseen Test N=951):
    1. CLIP ViT-B/16 Zero-Shot (Basic Prompts, Multi-View RGB)
    2. CLIP ViT-B/16 Zero-Shot (Physics Prompt Ensemble, Multi-View RGB)
    3. CLIP ViT-B/16 Zero-Shot (Physics Prompt Ensemble, 1.0s Viridis Colormap)
- Part B (Frozen CLIP Visual Representations + Lightweight Probes, Pooled Seed 42):
    - Extracts & saves frozen 512-dim CLIP embeddings for ALL 6,340 glitches (`clip_embeddings_frozen.npz`)
    4. CLIP ViT-B/16 Frozen Linear Probe (Tuned Multinomial Logistic Regression)
    5. CLIP ViT-B/16 Frozen MLP Probe (2-Layer PyTorch MLP with smoothed class weights)
- Part C1 (Prompt-Initialized End-to-End Fine-Tuned CLIP ViT-B/16, Pooled Seed 42):
    6. CLIP ViT-B/16 Fine-Tuned (Text-Init, Multi-View RGB) + saves fine-tuned embeddings (`clip_embeddings_finetuned.npz`)
- Part C2 (5-Seed Cross-Detector Transfer & Few-Shot Adaptation H1 <-> L1 on Frozen CLIP Embeddings):
    - Evaluates 0% Text-Only Zero-Shot, 0% Source-Trained Probe, 5%/10%/20% Few-Shot Target-Adapted Probes,
      and 100% Native Target Probe across seeds [42..46].

Saves all tables, 3 publication-grade figures, both lightweight fp16 embedding archives (~6.5 MB each)
for local Step 17 Multimodal Fusion, and packs `step16_results.zip`.
"""

import argparse
import copy
import json
import os
import sys
import time
import zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import get_class_mapping
from src.data.split_utils import create_stratified_glitch_split
from src.models.cnn import compute_class_weights
from src.models.vlm_models import (
    CLIP_MODEL_ID,
    CLIPVisionTextWrapper,
    CLIPTextInitializedAdapter,
    FrozenCLIPMLPProbe,
    grayscale_to_viridis_rgb,
    count_trainable_parameters,
)
from src.experiments.run_transfer_learning import (
    SEEDS,
    FEW_SHOT_FRACTIONS,
    set_all_seeds,
    preload_multiview_rgb,
    build_tensor_arrays,
    sample_few_shot_stratified,
)
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


@torch.no_grad()
def extract_clip_embeddings(vlm_or_adapter, X_arr, device, batch_size=128, is_adapter=False):
    """Extracts L2-normalized 512-dim CLIP image embeddings cleanly on GPU with AMP."""
    vlm_or_adapter.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_arr)), batch_size=batch_size, shuffle=False)
    use_amp = (device.type == "cuda")
    embs = []
    for (bx,) in loader:
        bx = bx.to(device, dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            if is_adapter:
                e = vlm_or_adapter.extract_embeddings(bx)
            else:
                e = vlm_or_adapter.encode_images(bx, normalize=True)
        embs.append(e.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def compute_top_k_accuracy(probs_or_logits, y_true_idx, k=3):
    """Computes Top-K classification accuracy in [0, 1]."""
    k_eff = min(k, probs_or_logits.shape[1])
    topk = np.argsort(probs_or_logits, axis=1)[:, -k_eff:]
    hits = [int(y_true_idx[i] in topk[i]) for i in range(len(y_true_idx))]
    return float(np.mean(hits))


def evaluate_zero_shot_from_embeddings(img_embs, text_protos_np, y_true_idx, class_names, logit_scale=100.0):
    """Computes zero-shot logits, Top-1/Top-3 metrics, and per-class Top-3 recall."""
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    logits = logit_scale * (img_embs @ text_protos_np.T)
    logits_shift = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits_shift) / np.sum(np.exp(logits_shift), axis=1, keepdims=True)

    pred_idx = np.argmax(probs, axis=1)
    y_true_str = [idx_to_class[int(i)] for i in y_true_idx]
    y_pred_str = [idx_to_class[int(i)] for i in pred_idx]

    metrics = compute_all_metrics(y_true_str, y_pred_str, class_names=class_names)
    top3_acc = compute_top_k_accuracy(probs, y_true_idx, k=3)

    # Per-class Top-1 & Top-3 recall
    k_eff = min(3, probs.shape[1])
    top3_idx = np.argsort(probs, axis=1)[:, -k_eff:]
    pc_rows = []
    f1_map = dict(zip(metrics["per_class"]["Class"], metrics["per_class"]["F1-Score"]))
    for c_idx, c_name in enumerate(class_names):
        mask = (y_true_idx == c_idx)
        n_c = int(mask.sum())
        if n_c == 0:
            t1_rec, t3_rec = 0.0, 0.0
        else:
            t1_rec = float(np.mean(pred_idx[mask] == c_idx))
            t3_rec = float(np.mean([c_idx in top3_idx[j] for j in np.where(mask)[0]]))
        pc_rows.append({
            "Class": c_name,
            "Support": n_c,
            "Top1_F1": round(float(f1_map[c_name]), 4),
            "Top1_Recall": round(t1_rec, 4),
            "Top3_Recall": round(t3_rec, 4),
        })

    return y_pred_str, probs, metrics, top3_acc, pd.DataFrame(pc_rows)


def fit_tuned_linear_probe(X_tr, y_tr, X_va, y_va, class_names, seed=42, sample_weight=None,
                           c_grid=(0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)):
    """Tunes L2-regularized Multinomial Logistic Regression on frozen CLIP embeddings via Val Macro-F1."""
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_va_str = [idx_to_class[int(i)] for i in y_va]

    best_clf = None
    best_val_mf1 = -1.0
    best_cfg = {}

    cw_options = [None] if sample_weight is not None else [None, "balanced"]
    for c_val in c_grid:
        for cw in cw_options:
            clf = LogisticRegression(
                C=c_val,
                max_iter=1000,
                class_weight=cw,
                random_state=seed,
            )
            clf.fit(X_tr, y_tr, sample_weight=sample_weight)
            preds_idx = clf.predict(X_va)
            preds_str = [idx_to_class[int(i)] for i in preds_idx]
            val_m = compute_all_metrics(y_va_str, preds_str, class_names=class_names)
            if val_m["macro_f1"] > best_val_mf1:
                best_val_mf1 = val_m["macro_f1"]
                best_clf = clf
                best_cfg = {"C": c_val, "class_weight": str(cw)}

    return best_clf, float(best_val_mf1), best_cfg


def train_frozen_mlp_probe(X_tr, y_tr, X_va, y_va, class_names, device, seed=42, epochs=45, batch_size=64):
    """Trains a 2-layer PyTorch MLP Probe on frozen 512-dim CLIP embeddings."""
    set_all_seeds(seed)
    num_classes = len(class_names)
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    model = FrozenCLIPMLPProbe(input_dim=X_tr.shape[1], hidden_dim=256, num_classes=num_classes, dropout=0.3).to(device)

    class_weights = compute_class_weights(y_tr, num_classes=num_classes, power=0.5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr.astype(np.float32)), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=min(batch_size, max(2, len(X_tr))),
        shuffle=True,
    )
    X_va_t = torch.from_numpy(X_va.astype(np.float32)).to(device)
    y_va_str = [idx_to_class[int(i)] for i in y_va]

    best_val_mf1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for _ in range(1, epochs + 1):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_preds = torch.argmax(model(X_va_t), dim=1).cpu().numpy()
        va_str = [idx_to_class[int(i)] for i in va_preds]
        va_mf1 = compute_all_metrics(y_va_str, va_str, class_names=class_names)["macro_f1"]
        if va_mf1 > best_val_mf1:
            best_val_mf1 = va_mf1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, float(best_val_mf1), float(time.time() - t0)


def train_clip_text_init_adapter(
    adapter, X_tr, y_tr, X_va, y_va, class_names, device,
    seed=42, epochs=12, batch_size=32, backbone_lr=1.5e-5, head_lr=5e-4,
    weight_decay=1e-2, patience=4, verbose=True
):
    """End-to-end fine-tuning of Text-Initialized CLIP ViT-B/16 with CUDA AMP and early stopping."""
    set_all_seeds(seed)
    adapter = adapter.to(device)
    num_classes = len(class_names)
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    class_weights = compute_class_weights(y_tr, num_classes=num_classes, power=0.5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        adapter.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr, weight_decay=weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=min(batch_size, max(2, len(X_tr))),
        shuffle=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
        batch_size=64,
        shuffle=False,
    )

    best_val_mf1 = -1.0
    best_state = copy.deepcopy(adapter.state_dict())
    no_improve = 0
    history = []
    t_total = time.time()

    if verbose:
        print(
            f"  Fine-tuning Text-Init CLIP ViT-B/16 on {device} (AMP={use_amp}) | "
            f"Train={len(X_tr)}, Val={len(X_va)} | Trainable Params={count_trainable_parameters(adapter):,}"
        )

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        adapter.train()
        tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0

        for bx, by in tr_loader:
            bx = bx.to(device, dtype=torch.float32)
            by = by.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = adapter(bx)
                loss = criterion(logits, by)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
                optimizer.step()

            tr_loss_sum += float(loss.item()) * bx.size(0)
            tr_correct += (torch.argmax(logits.detach(), dim=1) == by).sum().item()
            tr_total += by.size(0)

        scheduler.step()
        train_loss = tr_loss_sum / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)

        adapter.eval()
        va_loss_sum, va_preds_all, va_true_all = 0.0, [], []
        with torch.no_grad():
            for bx, by in va_loader:
                bx = bx.to(device, dtype=torch.float32)
                by = by.to(device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits = adapter(bx)
                    loss = criterion(logits, by)
                va_loss_sum += float(loss.item()) * bx.size(0)
                va_preds_all.extend(torch.argmax(logits.float(), dim=1).cpu().numpy().tolist())
                va_true_all.extend(by.cpu().numpy().tolist())

        val_loss = va_loss_sum / max(1, len(va_true_all))
        va_metrics = compute_all_metrics(
            [idx_to_class[i] for i in va_true_all],
            [idx_to_class[i] for i in va_preds_all],
            class_names=class_names,
        )
        val_acc = va_metrics["accuracy"]
        val_mf1 = va_metrics["macro_f1"]
        ep_sec = time.time() - ep_t0

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc * 100, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc * 100, 2),
            "val_macro_f1": round(val_mf1, 4),
            "logit_scale": round(float(adapter.logit_scale.exp().clamp(max=100.0).item()), 2),
            "epoch_time_sec": round(ep_sec, 2),
        })

        star = ""
        if val_mf1 > best_val_mf1:
            best_val_mf1 = val_mf1
            best_state = copy.deepcopy(adapter.state_dict())
            no_improve = 0
            star = " *"
        else:
            no_improve += 1

        if verbose:
            print(
                f"    Ep {epoch:02d}/{epochs:02d} ({ep_sec:4.1f}s) | "
                f"Tr Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
                f"Val Acc: {val_acc*100:5.2f}% Macro-F1: {val_mf1:.4f} "
                f"(scale={history[-1]['logit_scale']:.1f}){star}"
            )

        if no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch} (patience={patience}).")
            break

    adapter.load_state_dict(best_state)
    return adapter, pd.DataFrame(history), float(best_val_mf1), float(time.time() - t_total)


@torch.no_grad()
def evaluate_adapter(adapter, X_eval, y_eval_idx, class_names, device, batch_size=64):
    """Evaluates fine-tuned CLIPTextInitializedAdapter and returns predictions, probs, and metrics."""
    adapter.eval()
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    loader = DataLoader(TensorDataset(torch.from_numpy(X_eval)), batch_size=batch_size, shuffle=False)
    use_amp = (device.type == "cuda")
    probs_list = []
    for (bx,) in loader:
        bx = bx.to(device, dtype=torch.float32)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = adapter(bx)
        probs_list.append(torch.softmax(logits.float(), dim=1).cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    pred_idx = np.argmax(probs, axis=1)
    y_true_str = [idx_to_class[int(i)] for i in y_eval_idx]
    y_pred_str = [idx_to_class[int(i)] for i in pred_idx]
    metrics = compute_all_metrics(y_true_str, y_pred_str, class_names=class_names)
    return y_pred_str, probs, metrics


def plot_vlm_benchmark(vlm_df, prior_bench_csv, save_path):
    """Plots unified benchmark comparing prior milestones (RF, 4-Layer CNN, ImageNet ViT) vs Step 16 CLIP paradigms."""
    vlm_names = set(vlm_df["Model"].tolist())
    keep_prior = {
        "Feature-Based Random Forest",
        "Adapted Gravity Spy CNN (1.0s)",
        "ResNet-18 (Multi-View RGB)",
        "ConvNeXt-Tiny (Multi-View RGB)",
        "ViT-B/16 (Multi-View RGB)",
    }
    rows = []
    if os.path.exists(prior_bench_csv):
        base_df = pd.read_csv(prior_bench_csv)
        for _, r in base_df.iterrows():
            if r["Model"] in keep_prior and r["Model"] not in vlm_names:
                rows.append({
                    "Model": r["Model"],
                    "Macro_F1": float(r["Macro-F1"]),
                    "Accuracy": float(r["Accuracy (%)"]),
                    "Group": "Prior",
                })
    for _, r in vlm_df.iterrows():
        rows.append({
            "Model": r["Model"],
            "Macro_F1": float(r["Macro-F1"]),
            "Accuracy": float(r["Accuracy (%)"]),
            "Group": "CLIP",
        })

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 6.8))
    colors = ["#7f7f7f" if g == "Prior" else "#9467bd" for g in plot_df["Group"]]
    bars = ax.barh(plot_df["Model"], plot_df["Macro_F1"], color=colors, edgecolor="black", alpha=0.88)

    for bar, acc in zip(bars, plot_df["Accuracy"]):
        w = bar.get_width()
        ax.text(
            w + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f} ({acc:.1f}% acc)",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0.0, 1.24)
    ax.set_xlabel("Unseen Test Macro-F1 Score (Primary Metric)", fontsize=11, fontweight="bold")
    ax.set_title("Step 16 Benchmark: Prior Vision & Feature Models vs. OpenAI CLIP ViT-B/16 Paradigms", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"VLM benchmark figure saved to: {save_path}")


def plot_zeroshot_and_cm(zs_detail_df, best_cm, class_names, best_title_label, best_macro_f1, save_path):
    """Plots 2-panel visual: (Left) Zero-Shot Top-1 vs Top-3 Recall per class, (Right) Best CLIP Confusion Matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), gridspec_kw={"width_ratios": [1.0, 1.25]})

    # Left panel: Per-class Zero-Shot Top-1 vs Top-3 Recall sorted by Top-3 Recall
    df_s = zs_detail_df.sort_values("Top3_Recall", ascending=True)
    y_pos = np.arange(len(df_s))
    ax1.barh(y_pos - 0.18, df_s["Top1_Recall"], height=0.35, color="#9467bd", edgecolor="black", label="Zero-Shot Top-1 Recall")
    ax1.barh(y_pos + 0.18, df_s["Top3_Recall"], height=0.35, color="#c5b0d5", edgecolor="black", label="Zero-Shot Top-3 Recall")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_s["Class"], fontsize=9.5, fontweight="bold")
    ax1.set_xlim(0.0, 1.12)
    ax1.set_xlabel("Recall (0 LIGO Training Images)", fontsize=10.5, fontweight="bold")
    ax1.set_title("Part A: True Zero-Shot CLIP Recall by Glitch Class\n(Physics Text Prompts Alone)", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(axis="x", linestyle="--", alpha=0.55)
    ax1.legend(loc="lower right", fontsize=9)

    # Right panel: Normalized Confusion Matrix for best fine-tuned CLIP
    cm_norm = best_cm.astype(float) / np.maximum(best_cm.sum(axis=1, keepdims=True), 1.0)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7.5},
        ax=ax2,
    )
    ax2.set_title(f"Part C: {best_title_label}\nNormalized Confusion Matrix (Macro-F1={best_macro_f1:.4f})", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=10.5, fontweight="bold")
    ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8.5)
    ax2.set_yticklabels(class_names, rotation=0, fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"VLM zero-shot + confusion matrix figure saved to: {save_path}")


def plot_clip_cross_detector_curve(clip_cd_summary, step13_cd_csv, step15_fs_csv, save_path):
    """Plots 2-panel 5-Seed Cross-Detector Few-Shot curves comparing Frozen CLIP Probe vs Step 13 & Step 15 models."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), sharey=True)

    stage_order = [
        "0% (Source Probe)",
        "5% Adapted Probe",
        "10% Adapted Probe",
        "20% Adapted Probe",
        "100% Native Target",
    ]
    step15_stage_map = {
        "0% (Zero-Shot)": 0,
        "5% Fine-Tuned": 1,
        "10% Fine-Tuned": 2,
        "20% Fine-Tuned": 3,
        "100% Native Target": 4,
    }
    x_pos = np.arange(len(stage_order))

    step13_refs = {}
    if os.path.exists(step13_cd_csv):
        s13 = pd.read_csv(step13_cd_csv)
        for _, r in s13.iterrows():
            step13_refs[(r["Model"], r["Direction"])] = float(r["Macro_F1_Mean"])

    step15_df = pd.read_csv(step15_fs_csv) if os.path.exists(step15_fs_csv) else None

    for ax, direction, title in [
        (axes[0], "H1->L1", "Hanford (H1) → Livingston (L1)"),
        (axes[1], "L1->H1", "Livingston (L1) → Hanford (H1)"),
    ]:
        sub_c = clip_cd_summary[(clip_cd_summary["Direction"] == direction) & (clip_cd_summary["Stage"].isin(stage_order))]
        sub_c = sub_c.set_index("Stage").loc[stage_order]
        c_means = sub_c["Macro_F1_Mean"].values
        c_stds = sub_c["Macro_F1_Std"].values

        ax.plot(x_pos, c_means, marker="s", linewidth=2.5, markersize=8, color="#9467bd", label="Step 16 Frozen CLIP Probe (0 Backbone FT)")
        ax.fill_between(x_pos, np.clip(c_means - c_stds, 0, 1), np.clip(c_means + c_stds, 0, 1), color="#9467bd", alpha=0.16)

        for xi, m, sd in zip(x_pos, c_means, c_stds):
            ax.text(xi, m - sd - 0.022, f"{m:.3f}", ha="center", va="top", fontsize=8.5, fontweight="bold", color="#5e3c99")

        if step15_df is not None:
            s15_d = step15_df[step15_df["Direction"] == direction].copy()
            s15_d["x"] = s15_d["Stage"].map(step15_stage_map)
            s15_d = s15_d.dropna(subset=["x"]).sort_values("x")
            if len(s15_d) == len(x_pos):
                ax.plot(
                    s15_d["x"].values,
                    s15_d["Macro_F1_Mean"].values,
                    marker="o",
                    linewidth=2.0,
                    linestyle="--",
                    color="#1f77b4",
                    label="Step 15 Fine-Tuned ResNet-18 (Ref)",
                )

        if ("Feature Random Forest", direction) in step13_refs:
            rf_v = step13_refs[("Feature Random Forest", direction)]
            ax.axhline(y=rf_v, color="#2ca02c", linestyle="--", linewidth=1.4, label=f"Step 13 Zero-Shot RF ({rf_v:.3f})")
        if ("Adapted Gravity Spy CNN", direction) in step13_refs:
            cnn_v = step13_refs[("Adapted Gravity Spy CNN", direction)]
            ax.axhline(y=cnn_v, color="#ff7f0e", linestyle=":", linewidth=1.7, label=f"Step 13 Zero-Shot 4-Layer CNN ({cnn_v:.3f})")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["0%\n(Source Only)", "5%\nTarget", "10%\nTarget", "20%\nTarget", "100%\nNative"], fontsize=9.5, fontweight="bold")
        ax.set_ylim(0.55, 1.03)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Labeled Target-Detector Glitches Used", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(loc="lower right", fontsize=8.2, frameon=True)

    axes[0].set_ylabel("Target-Site Test Macro-F1 (5-Seed Mean ± Std)", fontsize=11, fontweight="bold")
    plt.suptitle("Step 16 Cross-Detector Generalization: Frozen CLIP Embeddings vs. Fine-Tuned Vision Backbones", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"VLM cross-detector figure saved to: {save_path}")


def package_step16_results_zip(output_dir, zip_path, generated_rel_paths):
    """Packages all generated Step 16 artifacts cleanly into `step16_results.zip`."""
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_subpath in generated_rel_paths:
            src_abs = os.path.join(output_dir, rel_subpath)
            if os.path.exists(src_abs):
                arcname = os.path.join("outputs", rel_subpath).replace("\\", "/")
                zf.write(src_abs, arcname=arcname)
    size_mb = os.path.getsize(zip_path) / (1024 ** 2)
    print(f"\nPackaged Step 16 deliverables into: {zip_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Step 16 (Novelty 2): Vision-Language Models (CLIP ViT-B/16) on LIGO Glitches")
    parser.add_argument("--smoke-test", action="store_true", help="Fast local verification on tiny subset (uses mock CLIP if transformers unavailable).")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "outputs"), help="Output directory for tables/, figures/, models/.")
    parser.add_argument("--h5-path", type=str, default=os.path.join(project_root, "trainingsetv1d1.h5"), help="Path to trainingsetv1d1.h5.")
    parser.add_argument("--zip-path", type=str, default=os.path.join(project_root, "step16_results.zip"), help="Destination path for results zip archive.")
    parser.add_argument("--model-id", type=str, default=CLIP_MODEL_ID, help="Hugging Face CLIP model identifier.")
    args = parser.parse_args()

    meta_csv = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    prior_bench_csv = os.path.join(project_root, "outputs", "tables", "benchmark_comparison.csv")
    prior_cd_csv = os.path.join(project_root, "outputs", "tables", "cross_detector_summary.csv")
    step15_fs_csv = os.path.join(project_root, "outputs", "tables", "few_shot_adaptation_summary.csv")

    tables_dir = os.path.join(args.output_dir, "tables")
    figures_dir = os.path.join(args.output_dir, "figures")
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_mock = False
    if args.smoke_test:
        try:
            import transformers  # noqa: F401
            # Still use mock during local smoke test to avoid downloading 600 MB weights offline
            use_mock = True
        except ImportError:
            use_mock = True

    print("=" * 84)
    print(" STEP 16 (NOVELTY 2): VISION-LANGUAGE MODELS — OPENAI CLIP ViT-B/16 ON LIGO GLITCHES")
    print("=" * 84)
    print(f"Mode       : {'SMOKE TEST (offline mock CLIP)' if args.smoke_test else 'FULL RUN (Real Pretrained CLIP)'}")
    print(f"Model ID   : {args.model_id}")
    print(f"Device     : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Output Dir : {args.output_dir}")

    meta_df = pd.read_csv(meta_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, _ = get_class_mapping(class_names)
    default_split = split_registry["default_split"]

    if args.smoke_test:
        smoke_parts = []
        for (_, _), grp in meta_df.groupby(["label", "ifo"]):
            smoke_parts.append(grp.head(10))
        work_meta = pd.concat(smoke_parts, ignore_index=True)

        split_s = create_stratified_glitch_split(work_meta, test_size=0.20, val_size=0.15, random_state=42)
        tr_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["train_ids"]))].reset_index(drop=True)
        va_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["val_ids"]))].reset_index(drop=True)
        te_df = work_meta[work_meta["gravityspy_id"].isin(set(split_s["test_ids"]))].reset_index(drop=True)

        mlp_epochs = 3
        ft_epochs = 1
        ft_patience = 1
        part_c_seeds = [42]
        part_c_fracs = [0.10]
        c_grid = (1.0, 10.0)
    else:
        work_meta = meta_df.copy().reset_index(drop=True)
        tr_df = work_meta[work_meta["gravityspy_id"].isin(set(default_split["train_ids"]))].copy().reset_index(drop=True)
        va_df = work_meta[work_meta["gravityspy_id"].isin(set(default_split["val_ids"]))].copy().reset_index(drop=True)
        te_df = work_meta[work_meta["gravityspy_id"].isin(set(default_split["test_ids"]))].copy().reset_index(drop=True)

        mlp_epochs = 45
        ft_epochs = 12
        ft_patience = 4
        part_c_seeds = SEEDS
        part_c_fracs = FEW_SHOT_FRACTIONS
        c_grid = (0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0)

    # 1. Preload Multi-View RGB [0.5s, 1.0s, 2.0s] spectrograms once
    spec_dict = preload_multiview_rgb(args.h5_path, work_meta, views=("0.5s", "1.0s", "2.0s"))

    # 2. Load Pretrained CLIP ViT-B/16
    print(f"\nLoading CLIP Vision-Language Model ({args.model_id}, use_mock={use_mock})...")
    t_load = time.time()
    vlm = CLIPVisionTextWrapper(model_id=args.model_id, use_mock=use_mock).to(device)
    vlm.eval()
    print(f"CLIP loaded in {time.time() - t_load:.1f}s | Shared Embed Dim={vlm.embed_dim}")

    # Build L2-normalized class text prototypes for Basic and Physics-Enriched prompts
    basic_protos = vlm.build_class_text_prototypes(class_names, prompt_mode="basic", device=device)
    physics_protos = vlm.build_class_text_prototypes(class_names, prompt_mode="physics", device=device)
    basic_protos_np = basic_protos.cpu().numpy()
    physics_protos_np = physics_protos.cpu().numpy()
    clip_scale_val = float(vlm.get_logit_scale().item())
    print(f"Built text prototypes: Basic={basic_protos_np.shape}, Physics={physics_protos_np.shape}, logit_scale={clip_scale_val:.2f}")

    # =========================================================================
    # PART B0 (DONE EARLY FOR REUSE): EXTRACT FROZEN CLIP EMBEDDINGS FOR ALL GLITCHES
    # =========================================================================
    print("\n" + "=" * 84)
    print(" EXTRACTING FROZEN CLIP ViT-B/16 EMBEDDINGS (ALL GLITCHES, MULTI-VIEW RGB)")
    print("=" * 84)
    t_emb = time.time()
    X_all_rgb, y_all_idx = build_tensor_arrays(work_meta, spec_dict, class_to_idx, in_mode="rgb_3view")
    frozen_embs_all = extract_clip_embeddings(vlm, X_all_rgb, device=device, batch_size=128, is_adapter=False)
    print(f"Extracted frozen embeddings: shape={frozen_embs_all.shape} in {time.time() - t_emb:.1f}s")

    # Map gravityspy_id -> row index in work_meta for fast zero-copy lookup
    gid_to_row = {gid: idx for idx, gid in enumerate(work_meta["gravityspy_id"].values)}

    def slice_embeddings(sub_df, emb_matrix):
        idxs = [gid_to_row[g] for g in sub_df["gravityspy_id"].values]
        y_arr = np.array([class_to_idx[lbl] for lbl in sub_df["label"].values], dtype=np.int64)
        return emb_matrix[idxs], y_arr

    # Save frozen CLIP embeddings (float16 -> ~6.5 MB for all 6,340 glitches!) for Step 17 Fusion
    frozen_npz_path = os.path.join(models_dir, "clip_embeddings_frozen.npz")
    np.savez_compressed(
        frozen_npz_path,
        gravityspy_id=np.asarray(work_meta["gravityspy_id"].values, dtype=str),
        label=np.asarray(work_meta["label"].values, dtype=str),
        ifo=np.asarray(work_meta["ifo"].values, dtype=str),
        embeddings=frozen_embs_all.astype(np.float16),
    )
    print(f"Saved frozen CLIP embeddings archive: {frozen_npz_path} ({os.path.getsize(frozen_npz_path)/(1024**2):.2f} MB)")

    # Release huge full-dataset RGB array right away
    del X_all_rgb

    # =========================================================================
    # PART A: TRUE ZERO-SHOT CLASSIFICATION (0 LIGO TRAINING IMAGES, N=951)
    # =========================================================================
    print("\n" + "=" * 84)
    print(" PART A: TRUE ZERO-SHOT VLM CLASSIFICATION (0 TRAINING IMAGES, UNSEEN TEST SET)")
    print("=" * 84)

    E_tr_frozen, y_tr = slice_embeddings(tr_df, frozen_embs_all)
    E_va_frozen, y_va = slice_embeddings(va_df, frozen_embs_all)
    E_te_frozen, y_te = slice_embeddings(te_df, frozen_embs_all)

    # Also extract Viridis-colormapped 1.0s test embeddings for Zero-Shot colormap comparison
    X_te_1s, _ = build_tensor_arrays(te_df, spec_dict, class_to_idx, in_mode="1.0s")
    X_te_viridis = grayscale_to_viridis_rgb(X_te_1s)
    E_te_viridis = extract_clip_embeddings(vlm, X_te_viridis, device=device, batch_size=128, is_adapter=False)
    del X_te_1s, X_te_viridis

    pooled_results = []
    per_class_df = pd.DataFrame({"Class": class_names})

    zs_configs = [
        ("CLIP ViT-B/16 Zero-Shot (Basic Prompts, RGB)", E_te_frozen, basic_protos_np, "Zero-Shot (Basic)", "Multi-View RGB"),
        ("CLIP ViT-B/16 Zero-Shot (Physics Prompts, RGB)", E_te_frozen, physics_protos_np, "Zero-Shot (Physics)", "Multi-View RGB"),
        ("CLIP ViT-B/16 Zero-Shot (Physics Prompts, 1.0s Viridis)", E_te_viridis, physics_protos_np, "Zero-Shot (Physics)", "1.0s Viridis"),
    ]

    best_zs_macro = -1.0
    best_zs_detail_df = None
    best_zs_label = None

    for lbl_name, E_eval, protos_np, paradigm, in_view in zs_configs:
        t0 = time.time()
        _, _, m_zs, top3_acc, detail_df = evaluate_zero_shot_from_embeddings(
            E_eval, protos_np, y_te, class_names, logit_scale=clip_scale_val
        )
        t_el = time.time() - t0
        print(
            f"  [{lbl_name}] Top-1 Acc={m_zs['accuracy']*100:.2f}% | "
            f"Top-3 Acc={top3_acc*100:.2f}% | Macro-F1={m_zs['macro_f1']:.4f} | Weighted-F1={m_zs['weighted_f1']:.4f}"
        )

        pooled_results.append({
            "Model": lbl_name,
            "Paradigm": paradigm,
            "Input_View": in_view,
            "Trainable_Params": 0,
            "Val_Macro_F1": np.nan,
            "Accuracy (%)": round(m_zs["accuracy"] * 100, 2),
            "Top3_Accuracy (%)": round(top3_acc * 100, 2),
            "Macro-F1": round(m_zs["macro_f1"], 4),
            "Weighted-F1": round(m_zs["weighted_f1"], 4),
            "Train_Time_Sec": round(t_el, 2),
        })
        f1_map = dict(zip(m_zs["per_class"]["Class"], m_zs["per_class"]["F1-Score"]))
        per_class_df[lbl_name] = [round(float(f1_map[c]), 4) for c in class_names]

        if m_zs["macro_f1"] > best_zs_macro:
            best_zs_macro = m_zs["macro_f1"]
            best_zs_detail_df = detail_df.copy()
            best_zs_label = lbl_name

    best_zs_detail_df["Best_ZeroShot_Config"] = best_zs_label
    best_zs_detail_df.to_csv(os.path.join(tables_dir, "vlm_clip_zeroshot_details.csv"), index=False)
    print(f"\nBest Zero-Shot Config: {best_zs_label} (Macro-F1={best_zs_macro:.4f})")
    print(best_zs_detail_df[["Class", "Support", "Top1_F1", "Top1_Recall", "Top3_Recall"]].to_string(index=False))

    # =========================================================================
    # PART B: FROZEN CLIP EMBEDDINGS + LIGHTWEIGHT PROBES (POOLED SEED 42)
    # =========================================================================
    print("\n" + "=" * 84)
    print(" PART B: FROZEN CLIP EMBEDDING PROBES (0 BACKBONE FINE-TUNING, SEED 42)")
    print("=" * 84)
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    # B1: Tuned Linear Probe (Multinomial Logistic Regression)
    t_lp = time.time()
    lin_clf, lp_val_mf1, lp_cfg = fit_tuned_linear_probe(
        E_tr_frozen, y_tr, E_va_frozen, y_va, class_names, seed=42, c_grid=c_grid
    )
    t_lp_sec = time.time() - t_lp
    lp_probs = lin_clf.predict_proba(E_te_frozen)
    lp_pred_idx = np.argmax(lp_probs, axis=1)
    lp_metrics = compute_all_metrics(
        [idx_to_class[int(i)] for i in y_te],
        [idx_to_class[int(i)] for i in lp_pred_idx],
        class_names=class_names,
    )
    lp_top3 = compute_top_k_accuracy(lp_probs, y_te, k=3)
    lp_label = "CLIP ViT-B/16 Frozen Linear Probe"
    print(f"  Best Linear Probe cfg={lp_cfg} | Val Macro-F1={lp_val_mf1:.4f}")
    print_metrics_summary(lp_label, lp_metrics)

    pooled_results.append({
        "Model": lp_label,
        "Paradigm": "Frozen Linear Probe",
        "Input_View": "Multi-View RGB",
        "Trainable_Params": int(lin_clf.coef_.size + lin_clf.intercept_.size),
        "Val_Macro_F1": round(lp_val_mf1, 4),
        "Accuracy (%)": round(lp_metrics["accuracy"] * 100, 2),
        "Top3_Accuracy (%)": round(lp_top3 * 100, 2),
        "Macro-F1": round(lp_metrics["macro_f1"], 4),
        "Weighted-F1": round(lp_metrics["weighted_f1"], 4),
        "Train_Time_Sec": round(t_lp_sec, 2),
    })
    f1_map = dict(zip(lp_metrics["per_class"]["Class"], lp_metrics["per_class"]["F1-Score"]))
    per_class_df[lp_label] = [round(float(f1_map[c]), 4) for c in class_names]

    # B2: Frozen 2-Layer MLP Probe
    mlp_model, mlp_val_mf1, mlp_sec = train_frozen_mlp_probe(
        E_tr_frozen, y_tr, E_va_frozen, y_va, class_names, device=device, seed=42, epochs=mlp_epochs
    )
    mlp_model.eval()
    with torch.no_grad():
        mlp_logits = mlp_model(torch.from_numpy(E_te_frozen.astype(np.float32)).to(device))
        mlp_probs = torch.softmax(mlp_logits, dim=1).cpu().numpy()
    mlp_pred_idx = np.argmax(mlp_probs, axis=1)
    mlp_metrics = compute_all_metrics(
        [idx_to_class[int(i)] for i in y_te],
        [idx_to_class[int(i)] for i in mlp_pred_idx],
        class_names=class_names,
    )
    mlp_top3 = compute_top_k_accuracy(mlp_probs, y_te, k=3)
    mlp_label = "CLIP ViT-B/16 Frozen MLP Probe"
    print_metrics_summary(mlp_label, mlp_metrics)

    pooled_results.append({
        "Model": mlp_label,
        "Paradigm": "Frozen MLP Probe",
        "Input_View": "Multi-View RGB",
        "Trainable_Params": count_trainable_parameters(mlp_model),
        "Val_Macro_F1": round(mlp_val_mf1, 4),
        "Accuracy (%)": round(mlp_metrics["accuracy"] * 100, 2),
        "Top3_Accuracy (%)": round(mlp_top3 * 100, 2),
        "Macro-F1": round(mlp_metrics["macro_f1"], 4),
        "Weighted-F1": round(mlp_metrics["weighted_f1"], 4),
        "Train_Time_Sec": round(mlp_sec, 2),
    })
    f1_map = dict(zip(mlp_metrics["per_class"]["Class"], mlp_metrics["per_class"]["F1-Score"]))
    per_class_df[mlp_label] = [round(float(f1_map[c]), 4) for c in class_names]

    # =========================================================================
    # PART C1: TEXT-INITIALIZED FINE-TUNED CLIP ViT-B/16 (POOLED SEED 42)
    # =========================================================================
    print("\n" + "=" * 84)
    print(" PART C1: PROMPT-INITIALIZED END-TO-END FINE-TUNED CLIP ViT-B/16 (SEED 42)")
    print("=" * 84)

    X_tr_rgb, _ = build_tensor_arrays(tr_df, spec_dict, class_to_idx, in_mode="rgb_3view")
    X_va_rgb, _ = build_tensor_arrays(va_df, spec_dict, class_to_idx, in_mode="rgb_3view")
    X_te_rgb, _ = build_tensor_arrays(te_df, spec_dict, class_to_idx, in_mode="rgb_3view")

    adapter = CLIPTextInitializedAdapter(vlm, physics_protos, init_scale=25.0)
    adapter, ft_hist_df, ft_val_mf1, ft_sec = train_clip_text_init_adapter(
        adapter, X_tr_rgb, y_tr, X_va_rgb, y_va, class_names, device=device,
        seed=42, epochs=ft_epochs, batch_size=32, backbone_lr=1.5e-5, head_lr=5e-4,
        weight_decay=1e-2, patience=ft_patience, verbose=True
    )
    ft_hist_df.to_csv(os.path.join(tables_dir, "vlm_clip_finetune_history.csv"), index=False)

    ft_preds, ft_probs, ft_metrics = evaluate_adapter(adapter, X_te_rgb, y_te, class_names, device=device, batch_size=64)
    ft_top3 = compute_top_k_accuracy(ft_probs, y_te, k=3)
    ft_label = "CLIP ViT-B/16 Fine-Tuned (Text-Init, RGB)"
    print_metrics_summary(ft_label, ft_metrics)

    pooled_results.append({
        "Model": ft_label,
        "Paradigm": "Text-Init Fine-Tuned",
        "Input_View": "Multi-View RGB",
        "Trainable_Params": count_trainable_parameters(adapter),
        "Val_Macro_F1": round(ft_val_mf1, 4),
        "Accuracy (%)": round(ft_metrics["accuracy"] * 100, 2),
        "Top3_Accuracy (%)": round(ft_top3 * 100, 2),
        "Macro-F1": round(ft_metrics["macro_f1"], 4),
        "Weighted-F1": round(ft_metrics["weighted_f1"], 4),
        "Train_Time_Sec": round(ft_sec, 1),
    })
    f1_map = dict(zip(ft_metrics["per_class"]["Class"], ft_metrics["per_class"]["F1-Score"]))
    per_class_df[ft_label] = [round(float(f1_map[c]), 4) for c in class_names]

    # Save best CLIP test predictions (matching Step 15 schema)
    pd.DataFrame({
        "gravityspy_id": te_df["gravityspy_id"].values,
        "ifo": te_df["ifo"].values,
        "sample_type": te_df["sample_type"].values,
        "true_label": te_df["label"].values,
        "vlm_best_pred": ft_preds,
        "vlm_best_model": ft_label,
        "confidence": np.round(np.max(ft_probs, axis=1), 4),
        "correct": (te_df["label"].values == np.array(ft_preds)),
    }).to_csv(os.path.join(tables_dir, "vlm_clip_best_test_predictions.csv"), index=False)

    # Extract & save fine-tuned CLIP embeddings for ALL glitches (for Step 17 Multimodal Fusion)
    print("\nExtracting fine-tuned CLIP 512-dim embeddings for all glitches (for Step 17 Fusion)...")
    del X_tr_rgb, X_va_rgb, X_te_rgb
    X_all_rgb2, _ = build_tensor_arrays(work_meta, spec_dict, class_to_idx, in_mode="rgb_3view")
    ft_embs_all = extract_clip_embeddings(adapter, X_all_rgb2, device=device, batch_size=128, is_adapter=True)
    del X_all_rgb2
    ft_npz_path = os.path.join(models_dir, "clip_embeddings_finetuned.npz")
    np.savez_compressed(
        ft_npz_path,
        gravityspy_id=np.asarray(work_meta["gravityspy_id"].values, dtype=str),
        label=np.asarray(work_meta["label"].values, dtype=str),
        ifo=np.asarray(work_meta["ifo"].values, dtype=str),
        embeddings=ft_embs_all.astype(np.float16),
    )
    print(f"Saved fine-tuned CLIP embeddings archive: {ft_npz_path} ({os.path.getsize(ft_npz_path)/(1024**2):.2f} MB)")

    del adapter, vlm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pooled_df = pd.DataFrame(pooled_results)
    pooled_df.to_csv(os.path.join(tables_dir, "vlm_clip_pooled_results.csv"), index=False)
    per_class_df.to_csv(os.path.join(tables_dir, "vlm_clip_per_class_f1.csv"), index=False)

    print("\n=== STEP 16 POOLED LEADERBOARD (CLIP ViT-B/16 PARADIGMS) ===")
    print(pooled_df[["Model", "Accuracy (%)", "Top3_Accuracy (%)", "Macro-F1", "Weighted-F1", "Train_Time_Sec"]].to_string(index=False))

    plot_vlm_benchmark(pooled_df, prior_bench_csv, os.path.join(figures_dir, "vlm_clip_benchmark.png"))
    plot_zeroshot_and_cm(
        best_zs_detail_df,
        ft_metrics["confusion_matrix"],
        class_names,
        ft_label,
        ft_metrics["macro_f1"],
        os.path.join(figures_dir, "vlm_clip_zeroshot_and_cm.png"),
    )

    # =========================================================================
    # PART C2: 5-SEED CROSS-DETECTOR TRANSFER & FEW-SHOT ADAPTATION (FROZEN CLIP)
    # =========================================================================
    print("\n" + "=" * 84)
    print(f" PART C2: 5-SEED CROSS-DETECTOR FEW-SHOT ON FROZEN CLIP EMBEDDINGS (SEEDS={part_c_seeds})")
    print("=" * 84)

    h1_meta = work_meta[work_meta["ifo"] == "H1"].copy().reset_index(drop=True)
    l1_meta = work_meta[work_meta["ifo"] == "L1"].copy().reset_index(drop=True)

    stage_names = [
        "0% (Text Zero-Shot)",
        "0% (Source Probe)",
        "5% Adapted Probe",
        "10% Adapted Probe",
        "20% Adapted Probe",
        "100% Native Target",
    ]
    if args.smoke_test:
        stage_names = ["0% (Text Zero-Shot)", "0% (Source Probe)", "10% Adapted Probe", "100% Native Target"]

    cd_records = []
    pc_accum = {
        (d, st): {c: [] for c in class_names}
        for d in ["H1->L1", "L1->H1"]
        for st in stage_names
    }
    cd_c_grid = (1.0, 10.0, 50.0) if not args.smoke_test else (1.0, 10.0)
    tgt_weight_grid = (1.0, 3.0, 10.0) if not args.smoke_test else (3.0,)

    for seed in part_c_seeds:
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

        E_h1_tr, y_h1_tr = slice_embeddings(h1_tr, frozen_embs_all)
        E_h1_va, y_h1_va = slice_embeddings(h1_va, frozen_embs_all)
        E_h1_te, y_h1_te = slice_embeddings(h1_te, frozen_embs_all)

        E_l1_tr, y_l1_tr = slice_embeddings(l1_tr, frozen_embs_all)
        E_l1_va, y_l1_va = slice_embeddings(l1_va, frozen_embs_all)
        E_l1_te, y_l1_te = slice_embeddings(l1_te, frozen_embs_all)

        # 1. Pure Text Zero-Shot on Target Test Sets (0 labels from ANY detector)
        _, _, m_txt_l1, _, _ = evaluate_zero_shot_from_embeddings(E_l1_te, physics_protos_np, y_l1_te, class_names, clip_scale_val)
        _, _, m_txt_h1, _, _ = evaluate_zero_shot_from_embeddings(E_h1_te, physics_protos_np, y_h1_te, class_names, clip_scale_val)

        # 2. Native Source Probes (100% H1 and 100% L1)
        clf_h1, _, _ = fit_tuned_linear_probe(E_h1_tr, y_h1_tr, E_h1_va, y_h1_va, class_names, seed=seed, c_grid=cd_c_grid)
        clf_l1, _, _ = fit_tuned_linear_probe(E_l1_tr, y_l1_tr, E_l1_va, y_l1_va, class_names, seed=seed, c_grid=cd_c_grid)

        def eval_clf(clf, X_e, y_e):
            p = clf.predict(X_e)
            return compute_all_metrics(
                [idx_to_class[int(i)] for i in y_e],
                [idx_to_class[int(i)] for i in p],
                class_names=class_names,
            )

        m_native_l1 = eval_clf(clf_l1, E_l1_te, y_l1_te)
        m_native_h1 = eval_clf(clf_h1, E_h1_te, y_h1_te)
        m_src_h1_to_l1 = eval_clf(clf_h1, E_l1_te, y_l1_te)
        m_src_l1_to_h1 = eval_clf(clf_l1, E_h1_te, y_h1_te)

        for direction, stage_lbl, n_tgt, m_res in [
            ("H1->L1", "0% (Text Zero-Shot)", 0, m_txt_l1),
            ("H1->L1", "0% (Source Probe)", 0, m_src_h1_to_l1),
            ("H1->L1", "100% Native Target", len(l1_tr), m_native_l1),
            ("L1->H1", "0% (Text Zero-Shot)", 0, m_txt_h1),
            ("L1->H1", "0% (Source Probe)", 0, m_src_l1_to_h1),
            ("L1->H1", "100% Native Target", len(h1_tr), m_native_h1),
        ]:
            cd_records.append({
                "Direction": direction, "Stage": stage_lbl, "Seed": seed,
                "Target_Train_Samples": n_tgt,
                "Accuracy (%)": round(m_res["accuracy"] * 100, 2),
                "Macro_F1": round(m_res["macro_f1"], 4),
                "Weighted_F1": round(m_res["weighted_f1"], 4),
            })
            for c_name, f1_v in zip(m_res["per_class"]["Class"], m_res["per_class"]["F1-Score"]):
                pc_accum[(direction, stage_lbl)][c_name].append(float(f1_v))

        # 3. Few-Shot Domain-Adapted Probes (Source + Upweighted Few-Shot Target, tuned on Target Val)
        for frac in part_c_fracs:
            stage_lbl = f"{int(round(frac * 100))}% Adapted Probe"

            for direction, E_src, y_src, tgt_tr_df, E_tva, y_tva, E_tte, y_tte in [
                ("H1->L1", E_h1_tr, y_h1_tr, l1_tr, E_l1_va, y_l1_va, E_l1_te, y_l1_te),
                ("L1->H1", E_l1_tr, y_l1_tr, h1_tr, E_h1_va, y_h1_va, E_h1_te, y_h1_te),
            ]:
                sub_tgt = sample_few_shot_stratified(tgt_tr_df, fraction=frac, seed=seed, min_per_class=2)
                E_sub, y_sub = slice_embeddings(sub_tgt, frozen_embs_all)

                X_comb = np.concatenate([E_src, E_sub], axis=0)
                y_comb = np.concatenate([y_src, y_sub], axis=0)

                best_adapt_clf = None
                best_adapt_val = -1.0
                for tw in tgt_weight_grid:
                    sw = np.concatenate([np.ones(len(y_src), dtype=np.float32), np.full(len(y_sub), tw, dtype=np.float32)])
                    clf_cand, val_mf1, _ = fit_tuned_linear_probe(
                        X_comb, y_comb, E_tva, y_tva, class_names, seed=seed, sample_weight=sw, c_grid=cd_c_grid
                    )
                    if val_mf1 > best_adapt_val:
                        best_adapt_val = val_mf1
                        best_adapt_clf = clf_cand

                m_ad = eval_clf(best_adapt_clf, E_tte, y_tte)
                cd_records.append({
                    "Direction": direction, "Stage": stage_lbl, "Seed": seed,
                    "Target_Train_Samples": len(sub_tgt),
                    "Accuracy (%)": round(m_ad["accuracy"] * 100, 2),
                    "Macro_F1": round(m_ad["macro_f1"], 4),
                    "Weighted_F1": round(m_ad["weighted_f1"], 4),
                })
                for c_name, f1_v in zip(m_ad["per_class"]["Class"], m_ad["per_class"]["F1-Score"]):
                    pc_accum[(direction, stage_lbl)][c_name].append(float(f1_v))

        print(
            f"  [Seed {seed}] H1->L1: SourceProbe={m_src_h1_to_l1['macro_f1']:.4f} (Native={m_native_l1['macro_f1']:.4f}) | "
            f"L1->H1: SourceProbe={m_src_l1_to_h1['macro_f1']:.4f} (Native={m_native_h1['macro_f1']:.4f})"
        )

    cd_seed_df = pd.DataFrame(cd_records)
    cd_seed_df.to_csv(os.path.join(tables_dir, "vlm_clip_cross_detector_per_seed.csv"), index=False)

    cd_sum_rows = []
    for direction in ["H1->L1", "L1->H1"]:
        sub_d = cd_seed_df[cd_seed_df["Direction"] == direction]
        base_src = float(sub_d[sub_d["Stage"] == "0% (Source Probe)"]["Macro_F1"].mean())
        for st in stage_names:
            sub_s = sub_d[sub_d["Stage"] == st]
            mf1_m = float(sub_s["Macro_F1"].mean())
            mf1_sd = float(sub_s["Macro_F1"].std(ddof=0))
            acc_m = float(sub_s["Accuracy (%)"].mean())
            acc_sd = float(sub_s["Accuracy (%)"].std(ddof=0))
            cd_sum_rows.append({
                "Direction": direction,
                "Stage": st,
                "Avg_Target_Train_Samples": int(round(sub_s["Target_Train_Samples"].mean())),
                "Accuracy_Mean (%)": round(acc_m, 2),
                "Accuracy_Std (%)": round(acc_sd, 2),
                "Macro_F1_Mean": round(mf1_m, 4),
                "Macro_F1_Std": round(mf1_sd, 4),
                "Gain_vs_SourceProbe": round(mf1_m - base_src, 4),
                "Num_Seeds": len(part_c_seeds),
            })

    cd_summary_df = pd.DataFrame(cd_sum_rows)
    cd_summary_df.to_csv(os.path.join(tables_dir, "vlm_clip_cross_detector_summary.csv"), index=False)

    pc_cd_rows = []
    for direction in ["H1->L1", "L1->H1"]:
        for c_name in class_names:
            r = {"Direction": direction, "Class": c_name}
            for st in stage_names:
                r[st] = round(float(np.mean(pc_accum[(direction, st)][c_name])), 4)
            pc_cd_rows.append(r)
    pd.DataFrame(pc_cd_rows).to_csv(os.path.join(tables_dir, "vlm_clip_cross_detector_per_class.csv"), index=False)

    print("\n=== PART C2 SUMMARY: 5-SEED CROSS-DETECTOR FROZEN CLIP PROBE ===")
    print(cd_summary_df.to_string(index=False))

    if not args.smoke_test:
        plot_clip_cross_detector_curve(
            cd_summary_df, prior_cd_csv, step15_fs_csv,
            os.path.join(figures_dir, "vlm_clip_cross_detector_curve.png"),
        )
    else:
        fig_sm, ax_sm = plt.subplots(figsize=(6, 4))
        for d in ["H1->L1", "L1->H1"]:
            sd = cd_summary_df[cd_summary_df["Direction"] == d]
            ax_sm.plot(sd["Stage"], sd["Macro_F1_Mean"], marker="s", label=d)
        ax_sm.set_title("Smoke CLIP Cross-Detector Plot")
        ax_sm.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "vlm_clip_cross_detector_curve.png"), dpi=120)
        plt.close(fig_sm)

    # Package all generated Step 16 deliverables into step16_results.zip
    step16_artifacts = [
        "tables/vlm_clip_pooled_results.csv",
        "tables/vlm_clip_per_class_f1.csv",
        "tables/vlm_clip_zeroshot_details.csv",
        "tables/vlm_clip_finetune_history.csv",
        "tables/vlm_clip_best_test_predictions.csv",
        "tables/vlm_clip_cross_detector_per_seed.csv",
        "tables/vlm_clip_cross_detector_summary.csv",
        "tables/vlm_clip_cross_detector_per_class.csv",
        "figures/vlm_clip_benchmark.png",
        "figures/vlm_clip_zeroshot_and_cm.png",
        "figures/vlm_clip_cross_detector_curve.png",
        "models/clip_embeddings_frozen.npz",
        "models/clip_embeddings_finetuned.npz",
    ]
    package_step16_results_zip(args.output_dir, args.zip_path, step16_artifacts)
    print("\nStep 16 execution finished successfully!")


if __name__ == "__main__":
    main()
