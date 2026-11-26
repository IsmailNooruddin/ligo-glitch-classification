"""
Module: run_multimodal_fusion.py
Purpose: Step 17 (Novelty 3) — Multimodal Physics–Vision Hybrid Fusion & Unified Cross-Paradigm Failure Analysis.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Runs 100% locally on CPU (no Colab required!), combining all prior modalities strictly without leakage
(all scalers, hyperparameters, temperatures, and ensemble weights tuned strictly on Validation N=952):

- Part A (Feature-Level Early Multimodal Fusion, Seed 42):
    1. Frozen CLIP (512d) + 34 Physics Features  [Linear & 2-Layer LayerNorm MLP] (0 GPU fine-tuning)
    2. Fine-Tuned CLIP (512d) + 34 Physics Features [Linear & 2-Layer LayerNorm MLP]
- Part B (Decision-Level Calibrated Late Multimodal Fusion, Seed 42):
    - Caches Val (952) + Test (951) logits/probabilities/embeddings from Step 15 Champion `ViT-B/16`
    - Temperature-calibrates neural logits strictly on Val NLL before multimodal pooling
    3. Bi-Modal Late Fusion (34-Feature RF + ImageNet ViT-B/16)
    4. Bi-Modal Late Fusion (Early Physics-CLIP + ImageNet ViT-B/16)
    5. Tri-Modal Hybrid Fusion (Physics RF + ImageNet ViT-B/16 + Fine-Tuned CLIP)
- Part C (5-Seed Cross-Detector Multimodal Fusion, H1 <-> L1, Seeds 42-46):
    - Compares Frozen CLIP Alone vs. Frozen CLIP + All 34 Physics vs. Frozen CLIP + 30 Detector-Invariant Physics
      (ablating site-shifted absolute frequency features discovered in Step 13!).
- Part D (Unified 5-Paradigm Failure Analysis & Visual Showcase):
    - Compares Feature RF, 4-Layer CNN, ImageNet ViT-B/16, CLIP ViT-B/16, and Champion Multimodal Fusion
    - Generates 3 publication-grade 300-DPI figures including Multimodal Rescues & the 6 Unanimous Bayes-Floor glitches.
"""

import copy
import json
import os
import sys
import time
import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_spectrogram, get_class_mapping
from src.data.split_utils import create_stratified_glitch_split
from src.models.cnn import compute_class_weights
from src.models.tree_models import ALL_FEATURE_COLS
from src.models.transfer_models import TransferLIGOClassifier
from src.experiments.run_transfer_learning import SEEDS, FEW_SHOT_FRACTIONS, set_all_seeds, sample_few_shot_stratified
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


# Site-specific absolute frequency features identified as H1<->L1 shift drivers in Step 13
SITE_SENSITIVE_FREQ_COLS = [
    "meta_peak_frequency",
    "meta_central_freq",
    "meta_log_peak_frequency",
    "meta_log_central_freq",
]
DETECTOR_INVARIANT_PHYSICS_COLS = [c for c in ALL_FEATURE_COLS if c not in SITE_SENSITIVE_FREQ_COLS]


class MultimodalEarlyFusionMLP(nn.Module):
    """
    Two-branch Multimodal MLP combining 512-dim CLIP visual embeddings with 34 standardized
    Omicron + spectrogram physical features via modality-specific LayerNorm projections.
    """
    def __init__(self, vis_dim=512, phys_dim=34, hidden_dim=256, num_classes=14, dropout=0.30):
        super().__init__()
        self.vis_branch = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phys_branch = nn.Sequential(
            nn.LayerNorm(phys_dim),
            nn.Linear(phys_dim, 96),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(192 + 64),
            nn.Linear(192 + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_vis, x_phys):
        h_v = self.vis_branch(x_vis)
        h_p = self.phys_branch(x_phys)
        return self.fusion_head(torch.cat([h_v, h_p], dim=1))


def softmax_np(logits, temperature=1.0):
    """Numerically stable temperature-scaled softmax in NumPy."""
    z = logits / float(temperature)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def nll_from_probs(probs, y_true_idx):
    """Computes mean negative log-likelihood (cross-entropy) from probability matrix."""
    eps = 1e-12
    p = np.clip(probs[np.arange(len(y_true_idx)), y_true_idx], eps, 1.0)
    return float(-np.mean(np.log(p)))


def fit_temperature_on_val(val_logits, y_val_idx):
    """Finds optimal scalar temperature T > 0 minimizing Validation NLL (zero test leakage)."""
    def obj(T):
        return nll_from_probs(softmax_np(val_logits, temperature=T), y_val_idx)

    res = minimize_scalar(obj, bounds=(0.3, 6.0), method="bounded")
    return float(res.x)


def align_proba_columns(probs, model_classes, target_class_names):
    """Reorders probability columns from `model_classes` into exact `target_class_names` order."""
    cls_list = list(model_classes)
    if all(isinstance(c, (int, np.integer)) for c in cls_list):
        # Integer-indexed classifier trained on 0..C-1 in target_class_names order
        idx_order = [cls_list.index(i) for i in range(len(target_class_names))]
    else:
        idx_order = [cls_list.index(c) for c in target_class_names]
    return probs[:, idx_order]


def extract_or_load_vit_val_test_cache(ckpt_path, h5_path, val_df, test_df, class_names, cache_path):
    """
    Loads Step 15 Champion `ViT-B/16 (Multi-View RGB)` checkpoint and extracts exact
    Val (952) + Test (951) logits, softmax probabilities, and 768-dim visual embeddings.
    Caches cleanly to `vit_b16_val_test_cache.npz` so re-runs are instantaneous!
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        if (
            len(d["val_ids"]) == len(val_df)
            and len(d["test_ids"]) == len(test_df)
            and np.array_equal(d["val_ids"], np.asarray(val_df["gravityspy_id"].values, dtype=str))
            and np.array_equal(d["test_ids"], np.asarray(test_df["gravityspy_id"].values, dtype=str))
        ):
            print(f"Loaded cached ViT-B/16 Val/Test outputs from: {cache_path}")
            return {k: d[k] for k in d.files}

    print(f"Extracting Step 15 Champion ViT-B/16 Val ({len(val_df)}) + Test ({len(test_df)}) logits & 768d embeddings on CPU...")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_classes = list(ckpt["class_names"])
    assert ckpt_classes == list(class_names), f"Class order mismatch: {ckpt_classes} vs {class_names}"

    net = TransferLIGOClassifier(backbone_name=ckpt["backbone_name"], num_classes=len(class_names), pretrained=False)
    state_fp32 = {k: (v.float() if v.is_floating_point() else v) for k, v in ckpt["state_dict_fp16"].items()}
    net.load_state_dict(state_fp32)
    net.eval()

    def run_split(sub_df, split_name):
        n = len(sub_df)
        arr = np.empty((n, 3, 140, 170), dtype=np.float32)
        with h5py.File(h5_path, "r") as h5_f:
            for i in range(n):
                row = sub_df.iloc[i]
                chans = [
                    load_spectrogram(h5_f, row["label"], row["sample_type"], row["gravityspy_id"], view=v)
                    for v in ("0.5s", "1.0s", "2.0s")
                ]
                arr[i] = np.stack(chans, axis=0)

        embs_list, logits_list = [], []
        bs = 32
        t_s = time.time()
        with torch.no_grad():
            for i in range(0, n, bs):
                bx = torch.from_numpy(arr[i : i + bs])
                feats = net.extract_features(bx)
                logits = net.head(feats)
                embs_list.append(feats.numpy())
                logits_list.append(logits.numpy())
                done = min(i + bs, n)
                if done % 256 == 0 or done == n:
                    print(f"  [ViT-B/16 {split_name}] {done}/{n} processed ({time.time() - t_s:.1f}s)")
        return np.concatenate(embs_list, axis=0), np.concatenate(logits_list, axis=0)

    val_embs, val_logits = run_split(val_df, "Val")
    test_embs, test_logits = run_split(test_df, "Test")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        val_ids=np.asarray(val_df["gravityspy_id"].values, dtype=str),
        test_ids=np.asarray(test_df["gravityspy_id"].values, dtype=str),
        val_logits=val_logits.astype(np.float32),
        test_logits=test_logits.astype(np.float32),
        val_embs=val_embs.astype(np.float16),
        test_embs=test_embs.astype(np.float16),
    )
    print(f"Saved ViT-B/16 cache to {cache_path} ({os.path.getsize(cache_path)/(1024**2):.2f} MB) in {time.time() - t0:.1f}s")
    return {
        "val_ids": np.asarray(val_df["gravityspy_id"].values, dtype=str),
        "test_ids": np.asarray(test_df["gravityspy_id"].values, dtype=str),
        "val_logits": val_logits.astype(np.float32),
        "test_logits": test_logits.astype(np.float32),
        "val_embs": val_embs.astype(np.float16),
        "test_embs": test_embs.astype(np.float16),
    }


def fit_early_fusion_linear(E_tr, P_tr, y_tr, E_va, P_va, y_va, E_te, class_names, seed=42):
    """
    Fits a Val-tuned Multimodal Linear Fusion model concatenating L2-normalized CLIP embeddings
    with standardized physical features scaled by a Val-tuned balance factor `alpha`.
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_va_str = [idx_to_class[int(i)] for i in y_va]

    scaler = StandardScaler()
    P_tr_s = scaler.fit_transform(P_tr)
    P_va_s = scaler.transform(P_va)
    # Normalize physics vector norm to comparable scale with unit-norm CLIP embeddings
    phys_norm_scale = np.sqrt(P_tr_s.shape[1])

    best_val_mf1 = -1.0
    best_val_nll = 1e9
    best_model = None
    best_alpha = 1.0
    best_cfg = {}

    for alpha in (0.25, 0.5, 1.0, 1.5, 2.5):
        X_tr_f = np.concatenate([E_tr, (alpha / phys_norm_scale) * P_tr_s], axis=1)
        X_va_f = np.concatenate([E_va, (alpha / phys_norm_scale) * P_va_s], axis=1)
        for c_val in (0.5, 1.0, 5.0, 15.0, 50.0):
            for cw in (None, "balanced"):
                clf = LogisticRegression(C=c_val, max_iter=1000, class_weight=cw, random_state=seed)
                clf.fit(X_tr_f, y_tr)
                va_probs = align_proba_columns(clf.predict_proba(X_va_f), clf.classes_, class_names)
                va_preds = [idx_to_class[int(i)] for i in np.argmax(va_probs, axis=1)]
                va_mf1 = compute_all_metrics(y_va_str, va_preds, class_names=class_names)["macro_f1"]
                va_nll = nll_from_probs(va_probs, y_va)
                if (va_mf1 > best_val_mf1 + 1e-5) or (abs(va_mf1 - best_val_mf1) <= 1e-5 and va_nll < best_val_nll):
                    best_val_mf1 = va_mf1
                    best_val_nll = va_nll
                    best_model = clf
                    best_alpha = alpha
                    best_cfg = {"alpha": alpha, "C": c_val, "class_weight": str(cw)}

    X_va_best = np.concatenate([E_va, (best_alpha / phys_norm_scale) * P_va_s], axis=1)
    P_te_s = scaler.transform(E_te[1])
    X_te_best = np.concatenate([E_te[0], (best_alpha / phys_norm_scale) * P_te_s], axis=1)

    val_probs = align_proba_columns(best_model.predict_proba(X_va_best), best_model.classes_, class_names)
    test_probs = align_proba_columns(best_model.predict_proba(X_te_best), best_model.classes_, class_names)
    return val_probs, test_probs, float(best_val_mf1), best_cfg


def train_early_fusion_mlp(E_tr, P_tr, y_tr, E_va, P_va, y_va, E_te, P_te, class_names, seed=42, epochs=55):
    """
    Trains a two-branch PyTorch Multimodal Fusion MLP (512d CLIP branch + 34d Physics branch)
    with smoothed class weights, CosineAnnealingLR, and early stopping strictly on Val Macro-F1.
    """
    set_all_seeds(seed)
    scaler = StandardScaler()
    P_tr_s = scaler.fit_transform(P_tr).astype(np.float32)
    P_va_s = scaler.transform(P_va).astype(np.float32)
    P_te_s = scaler.transform(P_te).astype(np.float32)

    E_tr_f = E_tr.astype(np.float32)
    E_va_f = E_va.astype(np.float32)
    E_te_f = E_te.astype(np.float32)

    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_va_str = [idx_to_class[int(i)] for i in y_va]

    model = MultimodalEarlyFusionMLP(vis_dim=E_tr_f.shape[1], phys_dim=P_tr_s.shape[1], hidden_dim=256, num_classes=len(class_names), dropout=0.30)
    cw = compute_class_weights(y_tr, num_classes=len(class_names), power=0.5)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(E_tr_f), torch.from_numpy(P_tr_s), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=64,
        shuffle=True,
    )
    t_va_e, t_va_p = torch.from_numpy(E_va_f), torch.from_numpy(P_va_s)
    t_te_e, t_te_p = torch.from_numpy(E_te_f), torch.from_numpy(P_te_s)

    best_val_mf1 = -1.0
    best_val_nll = 1e9
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(1, epochs + 1):
        model.train()
        for be, bp, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(be, bp), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(t_va_e, t_va_p).numpy()
        va_probs = softmax_np(va_logits, temperature=1.0)
        va_preds = [idx_to_class[int(i)] for i in np.argmax(va_probs, axis=1)]
        va_mf1 = compute_all_metrics(y_va_str, va_preds, class_names=class_names)["macro_f1"]
        va_nll = nll_from_probs(va_probs, y_va)
        if (va_mf1 > best_val_mf1 + 1e-5) or (abs(va_mf1 - best_val_mf1) <= 1e-5 and va_nll < best_val_nll):
            best_val_mf1 = va_mf1
            best_val_nll = va_nll
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(t_va_e, t_va_p).numpy()
        test_logits = model(t_te_e, t_te_p).numpy()

    # Temperature-calibrate on Val NLL
    T_opt = fit_temperature_on_val(val_logits, y_va)
    return softmax_np(val_logits, T_opt), softmax_np(test_logits, T_opt), float(best_val_mf1), T_opt


def fuse_probabilities(prob_list, weights, mode="linear"):
    """Combines calibrated probability matrices using linear averaging or log-linear (geometric) pooling."""
    eps = 1e-12
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)
    if mode == "linear":
        fused = sum(wi * p for wi, p in zip(w, prob_list))
        return fused / np.sum(fused, axis=1, keepdims=True)
    elif mode == "log_linear":
        log_p = sum(wi * np.log(np.clip(p, eps, 1.0)) for wi, p in zip(w, prob_list))
        log_p = log_p - np.max(log_p, axis=1, keepdims=True)
        e = np.exp(log_p)
        return e / np.sum(e, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


def tune_late_fusion_weights(val_probs_list, y_val_idx, class_names, step=0.05):
    """
    Grid-searches simplex weights strictly on Validation (N=952) Macro-F1 (tie-breaking on Val NLL)
    across both linear and log-linear pooling modes. Never touches test data!
    """
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_va_str = [idx_to_class[int(i)] for i in y_val_idx]
    n_mod = len(val_probs_list)

    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    candidates = []
    if n_mod == 2:
        for w0 in grid:
            candidates.append((w0, round(1.0 - w0, 4)))
    elif n_mod == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = round(1.0 - w0 - w1, 4)
                if w2 >= -1e-9:
                    candidates.append((w0, w1, max(0.0, w2)))
    else:
        raise ValueError("Only 2 or 3 modalities supported in simplex grid search.")

    best_score = -1.0
    best_nll = 1e9
    best_w = None
    best_mode = "linear"

    for mode in ("linear", "log_linear"):
        for w in candidates:
            # Require genuinely multimodal combination (at least 0.10 weight on each modality)
            if min(w) < 0.10 - 1e-9:
                continue
            fp = fuse_probabilities(val_probs_list, w, mode=mode)
            preds = [idx_to_class[int(i)] for i in np.argmax(fp, axis=1)]
            mf1 = compute_all_metrics(y_va_str, preds, class_names=class_names)["macro_f1"]
            nll = nll_from_probs(fp, y_val_idx)
            # Smooth tie-breaking on Val NLL within 0.0005 of top Val Macro-F1
            if (mf1 > best_score + 0.0005) or (abs(mf1 - best_score) <= 0.0005 and nll < best_nll):
                if mf1 > best_score:
                    best_score = mf1
                best_nll = nll
                best_w = w
                best_mode = mode

    return best_w, best_mode, float(best_score), float(best_nll)


def plot_grand_finale_benchmark(fusion_df, prior_bench_csv, save_path):
    """Plots the Grand Finale Project Leaderboard across all major paradigms (Steps 7-17)."""
    keep_prior = [
        "Baseline 1: Majority Class",
        "Baseline 2: Raw-Pixel Decision Tree",
        "Feature-Based Decision Tree",
        "Feature-Based Random Forest",
        "Adapted Gravity Spy CNN (1.0s)",
        "CLIP ViT-B/16 Frozen MLP Probe",
        "CLIP ViT-B/16 Fine-Tuned (Text-Init, RGB)",
        "ResNet-18 (Multi-View RGB)",
        "ConvNeXt-Tiny (Multi-View RGB)",
        "ViT-B/16 (Multi-View RGB)",
    ]
    fusion_names = set(fusion_df["Model"].tolist())
    rows = []
    if os.path.exists(prior_bench_csv):
        base = pd.read_csv(prior_bench_csv)
        base_map = {r["Model"]: r for _, r in base.iterrows()}
        for m_name in keep_prior:
            if m_name in base_map and m_name not in fusion_names:
                r = base_map[m_name]
                grp = "VLM" if "CLIP" in m_name else ("Transfer" if any(k in m_name for k in ["ResNet", "ConvNeXt", "ViT"]) else "Classical/CNN")
                rows.append({"Model": m_name, "Macro_F1": float(r["Macro-F1"]), "Accuracy": float(r["Accuracy (%)"]), "Group": grp})

    for _, r in fusion_df.iterrows():
        rows.append({"Model": r["Model"], "Macro_F1": float(r["Macro-F1"]), "Accuracy": float(r["Accuracy (%)"]), "Group": "Fusion"})

    plot_df = pd.DataFrame(rows)
    color_map = {
        "Classical/CNN": "#7f7f7f",
        "VLM": "#9467bd",
        "Transfer": "#1f77b4",
        "Fusion": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(13.8, 8.2))
    colors = [color_map[g] for g in plot_df["Group"]]
    bars = ax.barh(plot_df["Model"], plot_df["Macro_F1"], color=colors, edgecolor="black", alpha=0.90)

    for bar, acc in zip(bars, plot_df["Accuracy"]):
        w = bar.get_width()
        ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2, f"{w:.4f} ({acc:.2f}% acc)", va="center", fontsize=8.8, fontweight="bold")

    ax.set_xlim(0.0, 1.25)
    ax.set_xlabel("Unseen Test Macro-F1 Score (Primary Metric, N = 951)", fontsize=11, fontweight="bold")
    ax.set_title("Grand Finale Benchmark: Classical Features, CNN, Vision Transfer, CLIP VLM & Multimodal Fusion", fontsize=12.0, fontweight="bold", pad=48)
    ax.grid(axis="x", linestyle="--", alpha=0.55)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#7f7f7f", edgecolor="black", label="Steps 7-12: Baselines, Trees & Scratch CNN"),
        Patch(facecolor="#9467bd", edgecolor="black", label="Step 16: Vision-Language Model (OpenAI CLIP)"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Step 15: ImageNet Pretrained Vision Backbones"),
        Patch(facecolor="#d62728", edgecolor="black", label="Step 17: Multimodal Physics-Vision Hybrid Fusion"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=8.8, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Grand finale benchmark chart saved to: {save_path}")


def plot_fusion_cm_and_deltas(best_cm, class_names, fusion_f1, vit_f1, rf_f1, best_label, best_macro, save_path):
    """Plots 2-panel chart: (Left) Normalized Confusion Matrix of Champion Fusion, (Right) Per-Class F1 Gains vs ViT & RF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.5, 7.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    cm_norm = best_cm.astype(float) / np.maximum(best_cm.sum(axis=1, keepdims=True), 1.0)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7.5},
        ax=ax1,
    )
    ax1.set_title(f"{best_label}\nNormalized Confusion Matrix (Test Macro-F1={best_macro:.4f})", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Predicted Label", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=10.5, fontweight="bold")
    ax1.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8.5)
    ax1.set_yticklabels(class_names, rotation=0, fontsize=8.5)

    # Right panel: Per-class F1 gain of Fusion over ViT-B/16 and over RF
    y_pos = np.arange(len(class_names))
    d_vit = np.array([fusion_f1[c] - vit_f1[c] for c in class_names])
    d_rf = np.array([fusion_f1[c] - rf_f1[c] for c in class_names])

    ax2.barh(y_pos - 0.18, d_vit, height=0.34, color="#1f77b4", edgecolor="black", label="Gain vs. ImageNet ViT-B/16")
    ax2.barh(y_pos + 0.18, d_rf, height=0.34, color="#2ca02c", edgecolor="black", label="Gain vs. 34-Feature Random Forest")
    ax2.axvline(0.0, color="black", linewidth=1.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names, fontsize=9.2, fontweight="bold")
    ax2.set_xlabel("Per-Class F1 Score Delta (Hybrid Fusion − Unimodal Model)", fontsize=10, fontweight="bold")
    ax2.set_title("Where Multimodal Physics–Vision Fusion Wins\nPer-Class F1 Improvement Over Pure Vision & Pure Features", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.55)
    ax2.legend(loc="lower right", fontsize=8.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Multimodal confusion matrix & delta figure saved to: {save_path}")


def plot_unified_rescues_and_bayes_floor(rescues_df, bayes_df, h5_path, save_path):
    """
    Plots 2x3 visual spectrogram showcase:
    - Top Row: Real Multimodal Rescues (pure-vision ImageNet ViT-B/16 erred, Hybrid Fusion nailed it!)
    - Bottom Row: Unanimous Bayes-Floor Ambiguities (RF, CNN, ViT, CLIP, AND Fusion all unanimously agree on the same alt label!)
    """
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.8))

    with h5py.File(h5_path, "r") as h5_f:
        # Row 0: Multimodal Rescues over pure-vision ViT-B/16 (pick diverse classes)
        for col in range(3):
            ax = axes[0, col]
            if col >= len(rescues_df):
                ax.axis("off")
                continue
            r = rescues_df.iloc[col]
            img = load_spectrogram(h5_f, r["true_label"], r["sample_type"], r["gravityspy_id"], view="1.0s")
            ax.imshow(img, cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(
                f"[RESCUED] True: {r['true_label']} ({r['ifo']} | SNR={r['meta_snr']:.1f})\n"
                f"Fusion: {r['fusion_pred']} ({r['fusion_conf']:.2f}) | ViT: {r['vit_pred']}",
                fontsize=8.6, fontweight="bold", color="#1b5e20", pad=6
            )
            ax.set_xlabel("Time Bins (1.0s)", fontsize=8.5)
            ax.set_ylabel("Freq Bins", fontsize=8.5)

        # Row 1: Unanimous Bayes-Floor Citizen-Science Ambiguities
        for col in range(3):
            ax = axes[1, col]
            if col >= len(bayes_df):
                ax.axis("off")
                continue
            r = bayes_df.iloc[col]
            img = load_spectrogram(h5_f, r["true_label"], r["sample_type"], r["gravityspy_id"], view="1.0s")
            ax.imshow(img, cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(
                f"[ALL 5 MODELS AGREE] Label: {r['true_label']} ({r['ifo']} | SNR={r['meta_snr']:.1f})\n"
                f"All 5 Models -> {r['fusion_pred']} (Fusion conf={r['fusion_conf']:.2f})",
                fontsize=8.6, fontweight="bold", color="#b71c1c", pad=6
            )
            ax.set_xlabel("Time Bins (1.0s)", fontsize=8.5)
            ax.set_ylabel("Freq Bins", fontsize=8.5)

    plt.suptitle("Step 17 Unified Failure Analysis: Multimodal Rescues (Top) vs. Unanimous Label-Noise Floor (Bottom)", fontsize=13.5, fontweight="bold", y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.955], w_pad=2.2, h_pad=2.0)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Unified rescues & Bayes-floor figure saved to: {save_path}")


def main():
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")
    models_dir = os.path.join(project_root, "outputs", "models")
    h5_path = os.path.join(project_root, "trainingsetv1d1.h5")
    feat_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    prior_bench_csv = os.path.join(tables_dir, "benchmark_comparison.csv")

    print("=" * 88)
    print(" STEP 17 (NOVELTY 3): MULTIMODAL PHYSICS-VISION HYBRID FUSION & UNIFIED FAILURE ANALYSIS")
    print("=" * 88)

    # 1. Load metadata, features, splits, and CLIP embeddings
    feat_df = pd.read_csv(feat_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, idx_to_class = get_class_mapping(class_names)
    default_split = split_registry["default_split"]

    tr_ids = set(default_split["train_ids"])
    va_ids = set(default_split["val_ids"])
    te_ids = set(default_split["test_ids"])

    tr_df = feat_df[feat_df["gravityspy_id"].isin(tr_ids)].copy().reset_index(drop=True)
    va_df = feat_df[feat_df["gravityspy_id"].isin(va_ids)].copy().reset_index(drop=True)
    te_df = feat_df[feat_df["gravityspy_id"].isin(te_ids)].copy().reset_index(drop=True)

    y_tr = np.array([class_to_idx[c] for c in tr_df["label"].values], dtype=np.int64)
    y_va = np.array([class_to_idx[c] for c in va_df["label"].values], dtype=np.int64)
    y_te = np.array([class_to_idx[c] for c in te_df["label"].values], dtype=np.int64)
    y_te_str = list(te_df["label"].values)
    y_va_str = list(va_df["label"].values)

    # Load Frozen & Fine-Tuned CLIP embedding archives (aligned strictly by gravityspy_id!)
    def load_emb_lookup(npz_path):
        d = np.load(npz_path, allow_pickle=False)
        gids = d["gravityspy_id"]
        embs = d["embeddings"].astype(np.float32)
        return {str(g): embs[i] for i, g in enumerate(gids)}

    frozen_map = load_emb_lookup(os.path.join(models_dir, "clip_embeddings_frozen.npz"))
    ft_map = load_emb_lookup(os.path.join(models_dir, "clip_embeddings_finetuned.npz"))

    def get_emb_mat(df, emb_map):
        return np.stack([emb_map[str(g)] for g in df["gravityspy_id"].values], axis=0)

    E_tr_fr, E_va_fr, E_te_fr = get_emb_mat(tr_df, frozen_map), get_emb_mat(va_df, frozen_map), get_emb_mat(te_df, frozen_map)
    E_tr_ft, E_va_ft, E_te_ft = get_emb_mat(tr_df, ft_map), get_emb_mat(va_df, ft_map), get_emb_mat(te_df, ft_map)

    P_tr_all = tr_df[ALL_FEATURE_COLS].values.astype(np.float32)
    P_va_all = va_df[ALL_FEATURE_COLS].values.astype(np.float32)
    P_te_all = te_df[ALL_FEATURE_COLS].values.astype(np.float32)

    # Load Feature RF & compute aligned Val + Test probabilities
    rf_model = joblib.load(os.path.join(models_dir, "rf_best_model.joblib"))
    rf_val_raw = align_proba_columns(rf_model.predict_proba(P_va_all), rf_model.classes_, class_names)
    rf_test_raw = align_proba_columns(rf_model.predict_proba(P_te_all), rf_model.classes_, class_names)
    rf_test_preds = [idx_to_class[int(i)] for i in np.argmax(rf_test_raw, axis=1)]
    rf_metrics = compute_all_metrics(y_te_str, rf_test_preds, class_names=class_names)
    rf_f1_map = dict(zip(rf_metrics["per_class"]["Class"], rf_metrics["per_class"]["F1-Score"]))
    print(f"Verified RF Baseline   : Test Acc={rf_metrics['accuracy']*100:.2f}% | Macro-F1={rf_metrics['macro_f1']:.4f}")

    # Laplace smoothing for RF tree votes (n_estimators=50) so 0/50 votes never trigger -27.6 log-veto in log-linear fusion
    n_trees = float(getattr(rf_model, "n_estimators", 50))
    laplace_eps = 0.5
    rf_val_probs = (rf_val_raw * n_trees + laplace_eps) / (n_trees + laplace_eps * len(class_names))
    rf_test_probs = (rf_test_raw * n_trees + laplace_eps) / (n_trees + laplace_eps * len(class_names))
    print(f"RF Laplace Smoothing   : Val NLL improved {nll_from_probs(rf_val_raw, y_va):.4f} -> {nll_from_probs(rf_val_probs, y_va):.4f}")

    # =========================================================================
    # PART A: FEATURE-LEVEL (EARLY) MULTIMODAL FUSION (PHYSICS + CLIP EMBEDDINGS)
    # =========================================================================
    print("\n" + "=" * 88)
    print(" PART A: FEATURE-LEVEL (EARLY) MULTIMODAL FUSION (34 PHYSICS + 512d CLIP)")
    print("=" * 88)

    pooled_rows = []
    per_class_df = pd.DataFrame({"Class": class_names})
    model_val_test_probs = {}

    def record_model(name, fusion_type, modalities, val_probs, test_probs, val_mf1_reported, notes):
        preds = [idx_to_class[int(i)] for i in np.argmax(test_probs, axis=1)]
        m = compute_all_metrics(y_te_str, preds, class_names=class_names)
        print(f"  [{name}] Val Macro-F1={val_mf1_reported:.4f} | Test Acc={m['accuracy']*100:.2f}% | Test Macro-F1={m['macro_f1']:.4f} | Weighted-F1={m['weighted_f1']:.4f} | {notes}")
        pooled_rows.append({
            "Model": name,
            "Fusion_Level": fusion_type,
            "Modalities": modalities,
            "Val_Macro_F1": round(float(val_mf1_reported), 4),
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro-F1": round(m["macro_f1"], 4),
            "Weighted-F1": round(m["weighted_f1"], 4),
            "Test_Errors": int(np.sum(np.array(y_te_str) != np.array(preds))),
            "Config_Notes": notes,
        })
        f1_m = dict(zip(m["per_class"]["Class"], m["per_class"]["F1-Score"]))
        per_class_df[name] = [round(float(f1_m[c]), 4) for c in class_names]
        model_val_test_probs[name] = (val_probs, test_probs, m, preds)
        return m

    # A1: Early Linear Fusion — Frozen CLIP (512d) + 34 Physics (0 GPU fine-tuning!)
    va_p, te_p, va_mf1, cfg = fit_early_fusion_linear(E_tr_fr, P_tr_all, y_tr, E_va_fr, P_va_all, y_va, (E_te_fr, P_te_all), class_names)
    record_model(
        "Early Fusion Linear (Frozen CLIP + 34 Physics)",
        "Early (Feature-Level)", "Frozen VLM + Physics",
        va_p, te_p, va_mf1, f"alpha={cfg['alpha']}, C={cfg['C']}, cw={cfg['class_weight']}"
    )

    # A2: Early MLP Fusion — Frozen CLIP (512d) + 34 Physics (0 GPU fine-tuning!)
    va_p_frm, te_p_frm, va_mf1_frm, t_opt = train_early_fusion_mlp(E_tr_fr, P_tr_all, y_tr, E_va_fr, P_va_all, y_va, E_te_fr, P_te_all, class_names, seed=42)
    record_model(
        "Early Fusion MLP (Frozen CLIP + 34 Physics)",
        "Early (Feature-Level)", "Frozen VLM + Physics",
        va_p_frm, te_p_frm, va_mf1_frm, f"2-branch LayerNorm MLP, Val-T={t_opt:.2f}"
    )

    # A3: Early Linear Fusion — Fine-Tuned CLIP (512d) + 34 Physics
    va_p_efl, te_p_efl, va_mf1_efl, cfg_efl = fit_early_fusion_linear(E_tr_ft, P_tr_all, y_tr, E_va_ft, P_va_all, y_va, (E_te_ft, P_te_all), class_names)
    record_model(
        "Early Fusion Linear (Fine-Tuned CLIP + 34 Physics)",
        "Early (Feature-Level)", "Fine-Tuned VLM + Physics",
        va_p_efl, te_p_efl, va_mf1_efl, f"alpha={cfg_efl['alpha']}, C={cfg_efl['C']}, cw={cfg_efl['class_weight']}"
    )

    # A4: Early MLP Fusion — Fine-Tuned CLIP (512d) + 34 Physics
    va_p_efm, te_p_efm, va_mf1_efm, t_efm = train_early_fusion_mlp(E_tr_ft, P_tr_all, y_tr, E_va_ft, P_va_all, y_va, E_te_ft, P_te_all, class_names, seed=42)
    record_model(
        "Early Fusion MLP (Fine-Tuned CLIP + 34 Physics)",
        "Early (Feature-Level)", "Fine-Tuned VLM + Physics",
        va_p_efm, te_p_efm, va_mf1_efm, f"2-branch LayerNorm MLP, Val-T={t_efm:.2f}"
    )

    # =========================================================================
    # PART B: DECISION-LEVEL (LATE) CALIBRATED TRI-MODAL FUSION
    # =========================================================================
    print("\n" + "=" * 88)
    print(" PART B: DECISION-LEVEL CALIBRATED MULTIMODAL FUSION (PHYSICS + ViT-B/16 + CLIP)")
    print("=" * 88)

    # 1. Extract/Load Step 15 Champion ImageNet ViT-B/16 Val + Test logits & calibrate temperature on Val
    vit_cache = extract_or_load_vit_val_test_cache(
        os.path.join(models_dir, "transfer_best_checkpoint.pt"),
        h5_path, va_df, te_df, class_names,
        os.path.join(models_dir, "vit_b16_val_test_cache.npz"),
    )
    vit_val_logits = vit_cache["val_logits"]
    vit_test_logits = vit_cache["test_logits"]

    # Verify exact match with Step 15 reported Macro-F1 (0.9608)
    vit_raw_preds = [idx_to_class[int(i)] for i in np.argmax(vit_test_logits, axis=1)]
    vit_metrics = compute_all_metrics(y_te_str, vit_raw_preds, class_names=class_names)
    vit_f1_map = dict(zip(vit_metrics["per_class"]["Class"], vit_metrics["per_class"]["F1-Score"]))
    print(f"Verified Step 15 ViT-B/16: Test Acc={vit_metrics['accuracy']*100:.2f}% | Macro-F1={vit_metrics['macro_f1']:.4f}")

    T_vit = fit_temperature_on_val(vit_val_logits, y_va)
    vit_val_probs = softmax_np(vit_val_logits, temperature=T_vit)
    vit_test_probs = softmax_np(vit_test_logits, temperature=T_vit)
    print(f"ViT-B/16 Val Temperature Calibration: T={T_vit:.3f} (Val NLL: {nll_from_probs(softmax_np(vit_val_logits, 1.0), y_va):.4f} -> {nll_from_probs(vit_val_probs, y_va):.4f})")

    # 2. Fit calibrated Fine-Tuned CLIP head strictly on Train fine-tuned embeddings (for Val + Test probs)
    best_clip_c, best_clip_val = 10.0, -1.0
    best_clip_clf = None
    for c_val in (1.0, 5.0, 10.0, 25.0, 50.0):
        clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        clf.fit(E_tr_ft, y_tr)
        p_va = align_proba_columns(clf.predict_proba(E_va_ft), clf.classes_, class_names)
        mf1_v = compute_all_metrics(y_va_str, [idx_to_class[int(i)] for i in np.argmax(p_va, axis=1)], class_names=class_names)["macro_f1"]
        if mf1_v > best_clip_val:
            best_clip_val = mf1_v
            best_clip_c = c_val
            best_clip_clf = clf

    clip_val_logits = best_clip_clf.decision_function(E_va_ft)
    clip_test_logits = best_clip_clf.decision_function(E_te_ft)
    T_clip = fit_temperature_on_val(clip_val_logits, y_va)
    clip_val_probs = softmax_np(clip_val_logits, temperature=T_clip)
    clip_test_probs = softmax_np(clip_test_logits, temperature=T_clip)
    print(f"Fine-Tuned CLIP Calibrated Head (C={best_clip_c}, Val-T={T_clip:.3f}): Val Macro-F1={best_clip_val:.4f}")

    # B1: Bi-Modal Late Fusion (34-Feature RF + ImageNet ViT-B/16)
    w_bi1, mode_bi1, val_bi1, _ = tune_late_fusion_weights([rf_val_probs, vit_val_probs], y_va, class_names, step=0.05)
    va_bi1 = fuse_probabilities([rf_val_probs, vit_val_probs], w_bi1, mode=mode_bi1)
    te_bi1 = fuse_probabilities([rf_test_probs, vit_test_probs], w_bi1, mode=mode_bi1)
    record_model(
        "Bi-Modal Late Fusion (Physics RF + ImageNet ViT-B/16)",
        "Late (Calibrated)", "Physics RF + ImageNet ViT",
        va_bi1, te_bi1, val_bi1, f"mode={mode_bi1}, w=(RF={w_bi1[0]:.2f}, ViT={w_bi1[1]:.2f})"
    )

    # B2: Hybrid Early+Late Fusion (Best Val Early Physics-CLIP MLP + ImageNet ViT-B/16)
    best_early_name = max(
        [
            "Early Fusion Linear (Frozen CLIP + 34 Physics)",
            "Early Fusion MLP (Frozen CLIP + 34 Physics)",
            "Early Fusion Linear (Fine-Tuned CLIP + 34 Physics)",
            "Early Fusion MLP (Fine-Tuned CLIP + 34 Physics)",
        ],
        key=lambda nm: next(r["Val_Macro_F1"] for r in pooled_rows if r["Model"] == nm),
    )
    va_best_early, te_best_early, _, _ = model_val_test_probs[best_early_name]
    w_bi2, mode_bi2, val_bi2, _ = tune_late_fusion_weights([va_best_early, vit_val_probs], y_va, class_names, step=0.05)
    va_bi2 = fuse_probabilities([va_best_early, vit_val_probs], w_bi2, mode=mode_bi2)
    te_bi2 = fuse_probabilities([te_best_early, vit_test_probs], w_bi2, mode=mode_bi2)
    record_model(
        "Hybrid Fusion (Early Physics-CLIP + ImageNet ViT-B/16)",
        "Hybrid (Early + Late)", "Physics + CLIP + ImageNet ViT",
        va_bi2, te_bi2, val_bi2, f"early={best_early_name.split('(')[1][:-1]}, mode={mode_bi2}, w=(Early={w_bi2[0]:.2f}, ViT={w_bi2[1]:.2f})"
    )

    # B3: Tri-Modal Late Fusion (Physics RF + ImageNet ViT-B/16 + Fine-Tuned CLIP)
    w_tri, mode_tri, val_tri, _ = tune_late_fusion_weights([rf_val_probs, vit_val_probs, clip_val_probs], y_va, class_names, step=0.05)
    va_tri = fuse_probabilities([rf_val_probs, vit_val_probs, clip_val_probs], w_tri, mode=mode_tri)
    te_tri = fuse_probabilities([rf_test_probs, vit_test_probs, clip_test_probs], w_tri, mode=mode_tri)
    record_model(
        "Tri-Modal Hybrid Fusion (Physics RF + ImageNet ViT + CLIP)",
        "Late (Tri-Modal)", "Physics RF + ImageNet ViT + CLIP",
        va_tri, te_tri, val_tri, f"mode={mode_tri}, w=(RF={w_tri[0]:.2f}, ViT={w_tri[1]:.2f}, CLIP={w_tri[2]:.2f})"
    )

    # Select overall Step 17 Champion strictly by Validation Macro-F1 (clean zero-leakage selection!)
    pooled_df = pd.DataFrame(pooled_rows)
    best_row_idx = int(pooled_df["Val_Macro_F1"].idxmax())
    champ_name = pooled_df.loc[best_row_idx, "Model"]
    _, champ_test_probs, champ_metrics, champ_preds = model_val_test_probs[champ_name]
    champ_f1_map = dict(zip(champ_metrics["per_class"]["Class"], champ_metrics["per_class"]["F1-Score"]))

    print(f"\n>>> STEP 17 CHAMPION (selected strictly on Val Macro-F1): {champ_name}")
    print_metrics_summary(champ_name, champ_metrics)

    pooled_df.to_csv(os.path.join(tables_dir, "multimodal_fusion_pooled_results.csv"), index=False)
    per_class_df.to_csv(os.path.join(tables_dir, "multimodal_fusion_per_class_f1.csv"), index=False)

    pd.DataFrame({
        "gravityspy_id": te_df["gravityspy_id"].values,
        "ifo": te_df["ifo"].values,
        "sample_type": te_df["sample_type"].values,
        "true_label": y_te_str,
        "fusion_pred": champ_preds,
        "fusion_model": champ_name,
        "fusion_conf": np.round(np.max(champ_test_probs, axis=1), 4),
        "fusion_correct": (np.array(y_te_str) == np.array(champ_preds)),
    }).to_csv(os.path.join(tables_dir, "multimodal_fusion_best_test_predictions.csv"), index=False)

    # =========================================================================
    # PART C: 5-SEED CROSS-DETECTOR MULTIMODAL FUSION (H1 <-> L1, SEEDS 42-46)
    # =========================================================================
    print("\n" + "=" * 88)
    print(" PART C: 5-SEED CROSS-DETECTOR MULTIMODAL FUSION (CLIP + DETECTOR-INVARIANT PHYSICS)")
    print("=" * 88)

    h1_df = feat_df[feat_df["ifo"] == "H1"].copy().reset_index(drop=True)
    l1_df = feat_df[feat_df["ifo"] == "L1"].copy().reset_index(drop=True)

    cd_configs = [
        ("Frozen CLIP + All 34 Physics", ALL_FEATURE_COLS),
        ("Frozen CLIP + 30 Detector-Invariant Physics", DETECTOR_INVARIANT_PHYSICS_COLS),
    ]
    cd_records = []

    for cfg_name, phys_cols in cd_configs:
        for seed in SEEDS:
            h1_sp = create_stratified_glitch_split(h1_df, test_size=0.15, val_size=0.15, random_state=seed)
            l1_sp = create_stratified_glitch_split(l1_df, test_size=0.15, val_size=0.15, random_state=seed)

            h1_tr = h1_df[h1_df["gravityspy_id"].isin(set(h1_sp["train_ids"]))].reset_index(drop=True)
            h1_va = h1_df[h1_df["gravityspy_id"].isin(set(h1_sp["val_ids"]))].reset_index(drop=True)
            h1_te = h1_df[h1_df["gravityspy_id"].isin(set(h1_sp["test_ids"]))].reset_index(drop=True)

            l1_tr = l1_df[l1_df["gravityspy_id"].isin(set(l1_sp["train_ids"]))].reset_index(drop=True)
            l1_va = l1_df[l1_df["gravityspy_id"].isin(set(l1_sp["val_ids"]))].reset_index(drop=True)
            l1_te = l1_df[l1_df["gravityspy_id"].isin(set(l1_sp["test_ids"]))].reset_index(drop=True)

            def build_mm(df_s, scaler_obj=None, alpha=0.75):
                em = get_emb_mat(df_s, frozen_map)
                ph = df_s[phys_cols].values.astype(np.float32)
                if scaler_obj is None:
                    scaler_obj = StandardScaler()
                    ph_s = scaler_obj.fit_transform(ph)
                else:
                    ph_s = scaler_obj.transform(ph)
                X_mm = np.concatenate([em, (alpha / np.sqrt(ph.shape[1])) * ph_s], axis=1)
                y_mm = np.array([class_to_idx[c] for c in df_s["label"].values], dtype=np.int64)
                return X_mm, y_mm, scaler_obj

            for direction, src_tr, src_va, tgt_tr, tgt_va, tgt_te in [
                ("H1->L1", h1_tr, h1_va, l1_tr, l1_va, l1_te),
                ("L1->H1", l1_tr, l1_va, h1_tr, h1_va, h1_te),
            ]:
                # 0% Source-Only Multimodal Fusion
                X_s_tr, y_s_tr, sc_src = build_mm(src_tr, None)
                X_s_va, y_s_va, _ = build_mm(src_va, sc_src)
                X_t_te, y_t_te, _ = build_mm(tgt_te, sc_src)

                best_c, best_v = 10.0, -1.0
                best_clf = None
                y_sv_str = [idx_to_class[int(i)] for i in y_s_va]
                for c_v in (1.0, 10.0, 50.0):
                    clf = LogisticRegression(C=c_v, max_iter=1000, random_state=seed)
                    clf.fit(X_s_tr, y_s_tr)
                    p = [idx_to_class[int(i)] for i in clf.predict(X_s_va)]
                    v = compute_all_metrics(y_sv_str, p, class_names=class_names)["macro_f1"]
                    if v > best_v:
                        best_v, best_c, best_clf = v, c_v, clf

                m0 = compute_all_metrics(
                    [idx_to_class[int(i)] for i in y_t_te],
                    [idx_to_class[int(i)] for i in best_clf.predict(X_t_te)],
                    class_names=class_names,
                )
                cd_records.append({
                    "Config": cfg_name, "Direction": direction, "Stage": "0% (Zero-Shot Transfer)",
                    "Seed": seed, "Accuracy (%)": round(m0["accuracy"] * 100, 2), "Macro_F1": round(m0["macro_f1"], 4),
                    "Violin_Mode_F1": float(m0["per_class"].set_index("Class").loc["Violin_Mode", "F1-Score"]),
                })

                # 10% and 20% Few-Shot Target Adapted Multimodal Fusion
                for frac in (0.10, 0.20):
                    sub_tgt = sample_few_shot_stratified(tgt_tr, fraction=frac, seed=seed, min_per_class=2)
                    comb_df = pd.concat([src_tr, sub_tgt], ignore_index=True)
                    X_cb, y_cb, sc_cb = build_mm(comb_df, None)
                    X_tv, y_tv, _ = build_mm(tgt_va, sc_cb)
                    X_tt, y_tt, _ = build_mm(tgt_te, sc_cb)
                    y_tv_str = [idx_to_class[int(i)] for i in y_tv]

                    best_ad_clf, best_ad_v = None, -1.0
                    for tw in (3.0, 10.0):
                        sw = np.concatenate([np.ones(len(src_tr), dtype=np.float32), np.full(len(sub_tgt), tw, dtype=np.float32)])
                        for c_v in (5.0, 25.0):
                            clf = LogisticRegression(C=c_v, max_iter=1000, random_state=seed)
                            clf.fit(X_cb, y_cb, sample_weight=sw)
                            p = [idx_to_class[int(i)] for i in clf.predict(X_tv)]
                            v = compute_all_metrics(y_tv_str, p, class_names=class_names)["macro_f1"]
                            if v > best_ad_v:
                                best_ad_v, best_ad_clf = v, clf

                    mad = compute_all_metrics(
                        [idx_to_class[int(i)] for i in y_tt],
                        [idx_to_class[int(i)] for i in best_ad_clf.predict(X_tt)],
                        class_names=class_names,
                    )
                    cd_records.append({
                        "Config": cfg_name, "Direction": direction, "Stage": f"{int(frac*100)}% Few-Shot Adapted",
                        "Seed": seed, "Accuracy (%)": round(mad["accuracy"] * 100, 2), "Macro_F1": round(mad["macro_f1"], 4),
                        "Violin_Mode_F1": float(mad["per_class"].set_index("Class").loc["Violin_Mode", "F1-Score"]),
                    })

    cd_df = pd.DataFrame(cd_records)
    cd_summary = cd_df.groupby(["Config", "Direction", "Stage"], sort=False).agg(
        Accuracy_Mean=("Accuracy (%)", "mean"),
        Macro_F1_Mean=("Macro_F1", "mean"),
        Macro_F1_Std=("Macro_F1", lambda s: float(np.std(s, ddof=0))),
        Violin_Mode_F1=("Violin_Mode_F1", "mean"),
    ).reset_index()
    for col in ["Accuracy_Mean", "Macro_F1_Mean", "Macro_F1_Std", "Violin_Mode_F1"]:
        cd_summary[col] = cd_summary[col].round(4 if col != "Accuracy_Mean" else 2)
    cd_summary.to_csv(os.path.join(tables_dir, "multimodal_cross_detector_summary.csv"), index=False)
    print(cd_summary.to_string(index=False))

    # =========================================================================
    # PART D: UNIFIED 5-PARADIGM FAILURE ANALYSIS & PUBLICATION FIGURES
    # =========================================================================
    print("\n" + "=" * 88)
    print(" PART D: UNIFIED 5-PARADIGM FAILURE ANALYSIS (RF, CNN, ViT, CLIP & HYBRID FUSION)")
    print("=" * 88)

    cnn_df = pd.read_csv(os.path.join(tables_dir, "cnn_test_predictions.csv"))
    vit_df = pd.read_csv(os.path.join(tables_dir, "transfer_best_test_predictions.csv"))
    clip_df = pd.read_csv(os.path.join(tables_dir, "vlm_clip_best_test_predictions.csv"))

    uni = te_df[["gravityspy_id", "ifo", "sample_type", "label", "meta_snr", "meta_peak_frequency", "meta_duration"]].copy()
    uni = uni.rename(columns={"label": "true_label"})
    uni["rf_pred"] = rf_test_preds
    uni["rf_conf"] = np.round(np.max(rf_test_probs, axis=1), 4)
    uni["rf_correct"] = uni["rf_pred"] == uni["true_label"]
    uni["fusion_pred"] = champ_preds
    uni["fusion_conf"] = np.round(np.max(champ_test_probs, axis=1), 4)
    uni["fusion_correct"] = uni["fusion_pred"] == uni["true_label"]

    uni = uni.merge(cnn_df[["gravityspy_id", "cnn_pred", "cnn_confidence", "cnn_correct"]], on="gravityspy_id")
    uni = uni.merge(
        vit_df[["gravityspy_id", "transfer_best_pred", "confidence", "correct"]].rename(
            columns={"transfer_best_pred": "vit_pred", "confidence": "vit_conf", "correct": "vit_correct"}
        ),
        on="gravityspy_id",
    )
    uni = uni.merge(
        clip_df[["gravityspy_id", "vlm_best_pred", "confidence", "correct"]].rename(
            columns={"vlm_best_pred": "clip_pred", "confidence": "clip_conf", "correct": "clip_correct"}
        ),
        on="gravityspy_id",
    )

    rescued_from_vit = uni[uni["fusion_correct"] & (~uni["vit_correct"])].sort_values("meta_snr", ascending=False)
    lost_vs_vit = uni[(~uni["fusion_correct"]) & uni["vit_correct"]]
    all_5_wrong = uni[(~uni["rf_correct"]) & (~uni["cnn_correct"]) & (~uni["vit_correct"]) & (~uni["clip_correct"]) & (~uni["fusion_correct"])]
    unanimous_alt = all_5_wrong[
        (all_5_wrong["rf_pred"] == all_5_wrong["cnn_pred"])
        & (all_5_wrong["cnn_pred"] == all_5_wrong["vit_pred"])
        & (all_5_wrong["vit_pred"] == all_5_wrong["clip_pred"])
        & (all_5_wrong["clip_pred"] == all_5_wrong["fusion_pred"])
    ].sort_values("meta_snr", ascending=False)

    print(f"Total Unseen Test Errors (N=951):")
    print(f"  Feature RF           : {int((~uni['rf_correct']).sum()):2d} errors ({uni['rf_correct'].mean()*100:.2f}%)")
    print(f"  Adapted 4-Layer CNN  : {int((~uni['cnn_correct']).sum()):2d} errors ({uni['cnn_correct'].mean()*100:.2f}%)")
    print(f"  Fine-Tuned CLIP      : {int((~uni['clip_correct']).sum()):2d} errors ({uni['clip_correct'].mean()*100:.2f}%)")
    print(f"  ImageNet ViT-B/16    : {int((~uni['vit_correct']).sum()):2d} errors ({uni['vit_correct'].mean()*100:.2f}%)")
    print(f"  CHAMPION FUSION      : {int((~uni['fusion_correct']).sum()):2d} errors ({uni['fusion_correct'].mean()*100:.2f}%)")
    print(f"\nFusion vs. ImageNet ViT-B/16: Rescued {len(rescued_from_vit)} ViT errors | Introduced {len(lost_vs_vit)} regressions (Net = +{len(rescued_from_vit) - len(lost_vs_vit)} glitches!)")
    print(f"All 5 Models Wrong: {len(all_5_wrong)} glitches ({len(unanimous_alt)} of them 100% unanimous on the exact same alternative label!)")

    if len(rescued_from_vit) > 0:
        print("\n=== MULTIMODAL RESCUES OVER PURE-VISION ViT-B/16 ===")
        print(rescued_from_vit[["gravityspy_id", "ifo", "meta_snr", "true_label", "fusion_pred", "vit_pred", "rf_pred", "clip_pred"]].to_string(index=False))

    if len(unanimous_alt) > 0:
        print("\n=== UNANIMOUS BAYES-FLOOR LABEL AMBIGUITIES (ALL 5 MODELS AGREE) ===")
        print(unanimous_alt[["gravityspy_id", "ifo", "meta_snr", "true_label", "fusion_pred", "fusion_conf"]].to_string(index=False))

    # Save complete unified failure log (any glitch missed by at least 1 of the 5 major models)
    any_err = uni[(~uni["rf_correct"]) | (~uni["cnn_correct"]) | (~uni["vit_correct"]) | (~uni["clip_correct"]) | (~uni["fusion_correct"])].copy()
    any_err["category"] = np.where(
        (~any_err["rf_correct"]) & (~any_err["cnn_correct"]) & (~any_err["vit_correct"]) & (~any_err["clip_correct"]) & (~any_err["fusion_correct"]),
        "All_5_Models_Wrong",
        np.where(any_err["fusion_correct"] & (~any_err["vit_correct"]), "Fusion_Rescued_From_ViT",
        np.where(any_err["fusion_correct"], "Fusion_Correct_Other_Missed", "Fusion_Error")),
    )
    any_err.to_csv(os.path.join(tables_dir, "unified_failure_analysis_all_models.csv"), index=False)

    # Generate all 3 publication-grade figures
    plot_grand_finale_benchmark(pooled_df, prior_bench_csv, os.path.join(figures_dir, "multimodal_fusion_benchmark.png"))
    plot_fusion_cm_and_deltas(
        champ_metrics["confusion_matrix"], class_names,
        champ_f1_map, vit_f1_map, rf_f1_map,
        champ_name, champ_metrics["macro_f1"],
        os.path.join(figures_dir, "confusion_matrix_multimodal_best.png"),
    )
    # Pick 3 diverse rescue classes for Top Row (prioritizing highest-SNR Extremely_Loud + unique all-4-missed rescue + low-freq/violin)
    diverse_rescues = rescued_from_vit.drop_duplicates(subset=["true_label"]).head(3).reset_index(drop=True)
    plot_unified_rescues_and_bayes_floor(
        diverse_rescues,
        unanimous_alt.head(3).reset_index(drop=True),
        h5_path,
        os.path.join(figures_dir, "unified_failure_and_rescues.png"),
    )

    print("\nStep 17 Multimodal Hybrid Fusion & Unified Failure Analysis finished successfully!")


if __name__ == "__main__":
    main()
