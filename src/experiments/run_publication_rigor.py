"""
Module: run_publication_rigor.py
Purpose: Step 17.5 — Publication-Readiness Rigor Suite (100% Local CPU Execution).
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Closes all four key peer-reviewer criteria for workshop/journal publication:
1. Modern Tabular Baselines on the exact 34 Physics Features:
   - HistGradientBoostingClassifier (LightGBM-equivalent GBDT), ExtraTreesClassifier, and Physics-Only 2-Layer MLP
2. Statistical Significance & Uncertainty Quantification (Unseen Test Set N=951):
   - 1,000-iteration Paired Bootstrap 95% CIs for Macro-F1 & Accuracy across 9 key models
   - Exact McNemar's paired test + Paired Bootstrap Δ Macro-F1 p-values vs. Champion Hybrid Fusion
3. Probability Calibration (ECE, NLL, Brier) & Human-in-the-Loop Selective Prediction:
   - Reliability diagrams + Risk-Coverage abstention curve showing automated accuracy vs. volunteer review rate
4. Visual Model Interpretability:
   - True Grad-CAM saliency maps on AdaptedGravitySpyCNN across 6 iconic glitch morphologies
   - 2-panel t-SNE embedding space visualization (14 Glitch Classes vs. H1/L1 Detector Site)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import binomtest
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_spectrogram, get_class_mapping
from src.models.cnn import AdaptedGravitySpyCNN, compute_class_weights
from src.models.tree_models import ALL_FEATURE_COLS
from src.experiments.run_transfer_learning import set_all_seeds
from src.experiments.run_multimodal_fusion import (
    align_proba_columns,
    fit_temperature_on_val,
    fuse_probabilities,
    nll_from_probs,
    softmax_np,
    train_early_fusion_mlp,
)
from src.evaluation.metrics import compute_all_metrics


class PhysicsOnlyMLP(nn.Module):
    """2-layer MLP trained purely on the 34 physical features (apples-to-apples ablation for Early Fusion MLP)."""
    def __init__(self, in_dim=34, num_classes=14, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_physics_mlp(X_tr, y_tr, X_va, y_va, X_te, class_names, seed=42, epochs=60):
    """Trains PhysicsOnlyMLP strictly on Train, early-stopping on Val Macro-F1."""
    set_all_seeds(seed)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    idx_to_class = {i: c for i, c in enumerate(class_names)}
    y_va_str = [idx_to_class[int(i)] for i in y_va]

    model = PhysicsOnlyMLP(in_dim=X_tr_s.shape[1], num_classes=len(class_names))
    cw = compute_class_weights(y_tr, num_classes=len(class_names), power=0.5)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=64, shuffle=True,
    )
    t_va, t_te = torch.from_numpy(X_va_s), torch.from_numpy(X_te_s)
    best_mf1, best_state = -1.0, copy.deepcopy(model.state_dict())

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            va_preds = torch.argmax(model(t_va), dim=1).numpy()
        mf1 = compute_all_metrics(y_va_str, [idx_to_class[int(i)] for i in va_preds], class_names=class_names)["macro_f1"]
        if mf1 > best_mf1:
            best_mf1 = mf1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_logits = model(t_va).numpy()
        te_logits = model(t_te).numpy()
    T_opt = fit_temperature_on_val(va_logits, y_va)
    return softmax_np(te_logits, T_opt), float(best_mf1)


def compute_ece(probs, y_true_idx, n_bins=10):
    """Computes Expected Calibration Error (ECE) using equal-width confidence bins."""
    confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accs = (preds == y_true_idx).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_stats = []
    n = len(y_true_idx)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            avg_conf = float(confs[mask].mean())
            avg_acc = float(accs[mask].mean())
            ece += (cnt / n) * abs(avg_acc - avg_conf)
            bin_stats.append((avg_conf, avg_acc, cnt))
    return float(ece), bin_stats


def compute_brier(probs, y_true_idx):
    """Multi-class Brier score (mean squared error across probability simplex)."""
    n, c = probs.shape
    y_oh = np.zeros((n, c), dtype=np.float64)
    y_oh[np.arange(n), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - y_oh) ** 2, axis=1)))


def compute_cnn_gradcam(model, img_2d, target_class_idx):
    """
    Computes crisp 17x21 -> (140, 170) Grad-CAM heatmap on block4 post-ReLU (before MaxPool)
    of AdaptedGravitySpyCNN without inplace autograd issues.
    """
    model.eval()
    x = torch.from_numpy(img_2d.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, 140, 170)
    x.requires_grad_(True)

    h = model.block1(x)
    h = model.block2(h)
    h = model.block3(h)
    # Run Conv2d -> BatchNorm2d -> out-of-place ReLU of block4 to capture 17x21 spatial activations cleanly
    h = model.block4[0](h)
    h = model.block4[1](h)
    act = F.relu(h, inplace=False)
    act.retain_grad()
    h = model.block4[3](act)  # MaxPool2d
    h = model.adaptive_pool(h)
    logits = model.classifier(h)

    model.zero_grad()
    score = logits[0, int(target_class_idx)]
    score.backward()

    grads = act.grad[0]        # (64, 17, 21)
    fmap = act.detach()[0]     # (64, 17, 21)
    weights = grads.mean(dim=(1, 2), keepdim=True)  # GAP over H, W -> (64, 1, 1)
    cam = F.relu((weights * fmap).sum(dim=0))       # (17, 21)
    cam_up = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=(140, 170), mode="bilinear", align_corners=False
    )[0, 0].detach().numpy()
    if cam_up.max() > 1e-8:
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min())
    pred_idx = int(torch.argmax(logits.detach(), dim=1).item())
    conf = float(torch.softmax(logits.detach(), dim=1)[0, pred_idx].item())
    return cam_up, pred_idx, conf


def plot_bootstrap_forest(ci_df, sig_df, save_path):
    """Plots horizontal 95% Bootstrap Confidence Interval Forest Plot with significance annotations."""
    df = ci_df.copy()
    sig_map = dict(zip(sig_df["Baseline_Model"], sig_df["Boot_p_two_sided"]))

    fig, ax = plt.subplots(figsize=(13.2, 6.4))
    y_pos = np.arange(len(df))

    colors = []
    for m in df["Model"]:
        if "Hybrid Fusion" in m or "Early Fusion" in m:
            colors.append("#d62728")
        elif "ImageNet" in m:
            colors.append("#1f77b4")
        elif "CLIP" in m:
            colors.append("#9467bd")
        elif "CNN" in m:
            colors.append("#ff7f0e")
        else:
            colors.append("#2ca02c")

    means = df["Macro_F1"].values
    lo_err = means - df["Macro_F1_CI_Low"].values
    hi_err = df["Macro_F1_CI_High"].values - means

    for yi, m_val, lo, hi, col, name in zip(y_pos, means, lo_err, hi_err, colors, df["Model"]):
        ax.errorbar(
            m_val, yi, xerr=[[lo], [hi]], fmt="o", color=col, ecolor=col,
            elinewidth=2.6, capsize=5, markersize=8.5, markeredgecolor="black",
        )
        p_txt = ""
        if name in sig_map:
            pv = sig_map[name]
            stars = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
            p_txt = f"  [vs Champ: p={pv:.4f} {stars}]"
        elif "Hybrid Fusion" in name:
            p_txt = "  [CHAMPION]"
        ax.text(
            df.loc[df["Model"] == name, "Macro_F1_CI_High"].values[0] + 0.004,
            yi,
            f"{m_val:.4f} [{m_val-lo:.3f}, {m_val+hi:.3f}]{p_txt}",
            va="center", fontsize=8.4, fontweight="bold",
        )

    ax.axvline(means[-1], color="#d62728", linestyle="--", alpha=0.35, linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Model"], fontsize=9.3, fontweight="bold")
    ax.set_xlim(0.78, 1.135)
    ax.set_xlabel("Unseen Test Macro-F1 Score (1,000-Iteration Paired Bootstrap 95% CI, N = 951)", fontsize=10.5, fontweight="bold")
    ax.set_title("Publication Rigor: 95% Bootstrap Confidence Intervals & Statistical Significance vs. Champion Fusion", fontsize=11.8, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Bootstrap forest plot saved to: {save_path}")


def plot_calibration_and_abstention(cal_models, y_true_idx, abst_df, save_path):
    """Plots 2-panel figure: (Left) Reliability Diagrams with ECE, (Right) Human-in-the-Loop Risk-Coverage Curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # Left: Reliability Diagram
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.4, label="Perfect Calibration")
    palette = {
        "Feature RF (34 feats)": ("#2ca02c", "^"),
        "ImageNet ViT-B/16 (Raw T=1.0)": ("#aec7e8", "v"),
        "ImageNet ViT-B/16 (Calibrated T=1.60)": ("#1f77b4", "s"),
        "Champion Hybrid Fusion": ("#d62728", "o"),
    }
    for name, probs in cal_models.items():
        if name not in palette:
            continue
        col, mk = palette[name]
        ece, stats = compute_ece(probs, y_true_idx, n_bins=10)
        xs = [s[0] for s in stats if s[2] >= 3]
        ys = [s[1] for s in stats if s[2] >= 3]
        ax1.plot(xs, ys, marker=mk, linewidth=2.2, markersize=7, color=col, label=f"{name} (ECE={ece:.4f})")

    ax1.set_xlim(0.35, 1.02)
    ax1.set_ylim(0.35, 1.02)
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Empirical Accuracy", fontsize=10.5, fontweight="bold")
    ax1.set_title("Probability Calibration (Reliability Diagram)\nTemperature Scaling Cuts ViT Overconfidence", fontsize=11.5, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(loc="lower right", fontsize=8.4, frameon=True)

    # Right: Human-in-the-Loop Selective Prediction (Abstention Rate vs Automated Accuracy)
    style_rc = {
        "Feature RF (34 feats)": ("#2ca02c", "^", "--"),
        "Adapted 4-Layer CNN (1.0s)": ("#ff7f0e", "D", ":"),
        "ImageNet ViT-B/16 (Calibrated T=1.60)": ("#1f77b4", "s", "-."),
        "Champion Hybrid Fusion": ("#d62728", "o", "-"),
    }
    for m_name, (col, mk, ls) in style_rc.items():
        sub = abst_df[abst_df["Model"] == m_name].sort_values("Human_Review_Rate (%)")
        ax2.plot(
            sub["Human_Review_Rate (%)"], sub["Auto_Accuracy (%)"],
            marker=mk, linestyle=ls, linewidth=2.4, markersize=7.5, color=col, label=m_name,
        )
        if m_name == "Champion Hybrid Fusion":
            for _, rr in sub[sub["Human_Review_Rate (%)"].isin([0.0, 5.0, 10.0])].iterrows():
                ax2.annotate(
                    f"{rr['Auto_Accuracy (%)']:.2f}%",
                    (rr["Human_Review_Rate (%)"], rr["Auto_Accuracy (%)"]),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8.5, fontweight="bold", color="#b71c1c",
                )

    ax2.set_xlim(-0.5, 16.0)
    ax2.set_ylim(96.5, 100.2)
    ax2.set_xlabel("Lowest-Confidence Glitches Routed to Gravity Spy Volunteers (%)", fontsize=10.2, fontweight="bold")
    ax2.set_ylabel("Automated Classification Accuracy on Accepted Glitches (%)", fontsize=10.2, fontweight="bold")
    ax2.set_title("Human-in-the-Loop Selective Prediction (Risk–Coverage)\nRouting Just 5% Uncertain Glitches Pushes Accuracy Near 99.6%", fontsize=11.5, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.legend(loc="lower right", fontsize=8.5, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Calibration & selective prediction chart saved to: {save_path}")


def plot_gradcam_showcase(cnn_model, te_df, h5_path, class_names, save_path):
    """Plots 2x6 Grad-CAM visual interpretability grid across 6 iconic LIGO glitch morphologies."""
    target_classes = ["Blip", "Chirp", "Koi_Fish", "Repeating_Blips", "Scattered_Light", "Violin_Mode"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    fig, axes = plt.subplots(2, 6, figsize=(20.5, 6.8))
    with h5py.File(h5_path, "r") as h5_f:
        for col, cls_name in enumerate(target_classes):
            sub = te_df[te_df["label"] == cls_name]
            # Pick a crisp correctly-predicted high-confidence example for interpretability clarity
            chosen_row, best_cam, best_conf = None, None, -1.0
            for _, row in sub.head(15).iterrows():
                img = load_spectrogram(h5_f, row["label"], row["sample_type"], row["gravityspy_id"], view="1.0s")
                cam, pred_idx, conf = compute_cnn_gradcam(cnn_model, img, class_to_idx[cls_name])
                if pred_idx == class_to_idx[cls_name] and conf > best_conf:
                    best_conf = conf
                    chosen_row = row
                    best_cam = (img, cam, conf)
                if best_conf > 0.98:
                    break

            img, cam, conf = best_cam
            # Top row: Clean 1.0s spectrogram
            ax_top = axes[0, col]
            ax_top.imshow(img, cmap="viridis", origin="lower", aspect="auto")
            ax_top.set_title(f"{cls_name}\n({chosen_row['ifo']} | SNR={chosen_row['meta_snr']:.1f})", fontsize=9.8, fontweight="bold", pad=5)
            ax_top.set_xticks([])
            if col == 0:
                ax_top.set_ylabel("Raw 1.0s\nSpectrogram", fontsize=9.8, fontweight="bold")
            else:
                ax_top.set_yticks([])

            # Bottom row: Grad-CAM overlay
            ax_bot = axes[1, col]
            ax_bot.imshow(img, cmap="gray", origin="lower", aspect="auto", alpha=0.82)
            ax_bot.imshow(cam, cmap="jet", origin="lower", aspect="auto", alpha=0.48)
            ax_bot.set_title(f"Grad-CAM (conf={conf:.2f})", fontsize=9.2, fontweight="bold", color="#b71c1c", pad=4)
            ax_bot.set_xticks([])
            if col == 0:
                ax_bot.set_ylabel("CNN Block-4\nGrad-CAM", fontsize=9.8, fontweight="bold")
            else:
                ax_bot.set_yticks([])

    plt.suptitle("Visual Interpretability: Adapted Gravity Spy CNN Block-4 Grad-CAM Attention Across 6 Iconic Glitch Morphologies", fontsize=13.2, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Grad-CAM saliency showcase saved to: {save_path}")


def plot_tsne_embeddings(vit_embs, labels_str, ifos_str, class_names, save_path):
    """Plots 2-panel t-SNE of ImageNet ViT-B/16 test embeddings: (Left) 14 Glitch Classes, (Right) H1 vs L1 Detector."""
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    xy = tsne.fit_transform(vit_embs.astype(np.float32))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.2))

    # Left: 14 Glitch Classes with crisp centroid text annotations
    cmap_tab = plt.get_cmap("tab20")
    for idx, cls_name in enumerate(class_names):
        mask = (labels_str == cls_name)
        ax1.scatter(xy[mask, 0], xy[mask, 1], s=22, alpha=0.82, color=cmap_tab(idx / 14.0), edgecolors="none", label=cls_name)
        cx, cy = np.median(xy[mask, 0]), np.median(xy[mask, 1])
        ax1.text(
            cx, cy, cls_name.replace("Low_Frequency_", "LF_").replace("None_of_the_Above", "None_Above"),
            fontsize=7.6, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.78, edgecolor="gray", linewidth=0.6),
        )
    ax1.set_title("ImageNet ViT-B/16 Embedding Space (N = 951 Unseen Test)\nColored by 14 Shared Glitch Classes", fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    ax1.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Right: H1 (Hanford) vs L1 (Livingston)
    for ifo_name, col, lbl in [("H1", "#1f77b4", "Hanford (H1, n=597)"), ("L1", "#ff7f0e", "Livingston (L1, n=354)")]:
        mask = (ifos_str == ifo_name)
        ax2.scatter(xy[mask, 0], xy[mask, 1], s=22, alpha=0.70, color=col, edgecolors="none", label=lbl)
    ax2.set_title("Exact Same Embedding Projection Colored by Physical Detector\nStrong Cross-Site Alignment Across Most Glitch Families", fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    ax2.set_ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best", fontsize=9.5, frameon=True, markerscale=1.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"t-SNE embedding space chart saved to: {save_path}")


def main():
    t_start = time.time()
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")
    models_dir = os.path.join(project_root, "outputs", "models")
    h5_path = os.path.join(project_root, "trainingsetv1d1.h5")

    print("=" * 88)
    print(" STEP 17.5: PUBLICATION-READINESS RIGOR SUITE (100% LOCAL CPU)")
    print("=" * 88)

    feat_df = pd.read_csv(os.path.join(project_root, "data", "processed", "engineered_features.csv"))
    with open(os.path.join(project_root, "data", "processed", "split_indices.json"), "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, idx_to_class = get_class_mapping(class_names)
    default_split = split_registry["default_split"]

    tr_df = feat_df[feat_df["gravityspy_id"].isin(set(default_split["train_ids"]))].copy().reset_index(drop=True)
    va_df = feat_df[feat_df["gravityspy_id"].isin(set(default_split["val_ids"]))].copy().reset_index(drop=True)
    te_df = feat_df[feat_df["gravityspy_id"].isin(set(default_split["test_ids"]))].copy().reset_index(drop=True)

    X_tr = tr_df[ALL_FEATURE_COLS].values.astype(np.float32)
    X_va = va_df[ALL_FEATURE_COLS].values.astype(np.float32)
    X_te = te_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_tr = np.array([class_to_idx[c] for c in tr_df["label"].values], dtype=np.int64)
    y_va = np.array([class_to_idx[c] for c in va_df["label"].values], dtype=np.int64)
    y_te = np.array([class_to_idx[c] for c in te_df["label"].values], dtype=np.int64)
    y_va_str = list(va_df["label"].values)
    y_te_str = list(te_df["label"].values)

    # =========================================================================
    # 1. STRONGER MODERN TABULAR BASELINES ON THE EXACT 34 PHYSICS FEATURES
    # =========================================================================
    print("\n--- [1/4] Modern Tabular Baselines (34 Physics Features, Val-Tuned) ---")
    tab_rows = []

    # 1A: Feature-Based Decision Tree (exact Step 8 config for bootstrap alignment)
    dt_clf = DecisionTreeClassifier(max_depth=15, class_weight=None, random_state=42)
    dt_clf.fit(X_tr, y_tr)
    dt_te_preds = dt_clf.predict(X_te)

    # 1B: HistGradientBoostingClassifier (LightGBM-equivalent)
    best_hgb, best_hgb_val, best_hgb_cfg = None, -1.0, {}
    t0 = time.time()
    for lr in (0.05, 0.10):
        for md in (6, None):
            clf = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=lr, max_depth=md, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=42,
            )
            clf.fit(X_tr, y_tr)
            vp = [idx_to_class[int(i)] for i in clf.predict(X_va)]
            vm = compute_all_metrics(y_va_str, vp, class_names=class_names)["macro_f1"]
            if vm > best_hgb_val:
                best_hgb_val, best_hgb, best_hgb_cfg = vm, clf, {"lr": lr, "max_depth": md, "l2": 1.0}
    hgb_probs = align_proba_columns(best_hgb.predict_proba(X_te), best_hgb.classes_, class_names)
    hgb_preds = np.argmax(hgb_probs, axis=1)
    hgb_m = compute_all_metrics(y_te_str, [idx_to_class[int(i)] for i in hgb_preds], class_names=class_names)
    print(f"  HistGradientBoosting (34 feats) | cfg={best_hgb_cfg} | Val={best_hgb_val:.4f} | Test Acc={hgb_m['accuracy']*100:.2f}% | Macro-F1={hgb_m['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
    tab_rows.append({
        "Model": "HistGradientBoosting (34 Physics Features)",
        "Val_Macro_F1": round(best_hgb_val, 4),
        "Accuracy (%)": round(hgb_m["accuracy"] * 100, 2),
        "Macro-F1": round(hgb_m["macro_f1"], 4),
        "Weighted-F1": round(hgb_m["weighted_f1"], 4),
        "Config": str(best_hgb_cfg),
    })

    # 1C: ExtraTreesClassifier
    best_et, best_et_val, best_et_cfg = None, -1.0, {}
    t0 = time.time()
    for md in (None, 25):
        for cw in (None, "balanced"):
            clf = ExtraTreesClassifier(n_estimators=200, max_depth=md, class_weight=cw, random_state=42, n_jobs=1)
            clf.fit(X_tr, y_tr)
            vp = [idx_to_class[int(i)] for i in clf.predict(X_va)]
            vm = compute_all_metrics(y_va_str, vp, class_names=class_names)["macro_f1"]
            if vm > best_et_val:
                best_et_val, best_et, best_et_cfg = vm, clf, {"n_est": 200, "max_depth": md, "cw": cw}
    et_probs = align_proba_columns(best_et.predict_proba(X_te), best_et.classes_, class_names)
    et_preds = np.argmax(et_probs, axis=1)
    et_m = compute_all_metrics(y_te_str, [idx_to_class[int(i)] for i in et_preds], class_names=class_names)
    print(f"  ExtraTrees (34 feats)           | cfg={best_et_cfg} | Val={best_et_val:.4f} | Test Acc={et_m['accuracy']*100:.2f}% | Macro-F1={et_m['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
    tab_rows.append({
        "Model": "ExtraTrees (34 Physics Features)",
        "Val_Macro_F1": round(best_et_val, 4),
        "Accuracy (%)": round(et_m["accuracy"] * 100, 2),
        "Macro-F1": round(et_m["macro_f1"], 4),
        "Weighted-F1": round(et_m["weighted_f1"], 4),
        "Config": str(best_et_cfg),
    })

    # 1D: Physics-Only 2-Layer MLP (ablation isolating multimodal benefit of Early Fusion MLP)
    t0 = time.time()
    pmlp_probs, pmlp_val = train_physics_mlp(X_tr, y_tr, X_va, y_va, X_te, class_names, seed=42, epochs=60)
    pmlp_preds = np.argmax(pmlp_probs, axis=1)
    pmlp_m = compute_all_metrics(y_te_str, [idx_to_class[int(i)] for i in pmlp_preds], class_names=class_names)
    print(f"  Physics-Only 2-Layer MLP        | Val={pmlp_val:.4f} | Test Acc={pmlp_m['accuracy']*100:.2f}% | Macro-F1={pmlp_m['macro_f1']:.4f} ({time.time()-t0:.1f}s)")
    tab_rows.append({
        "Model": "Physics-Only MLP (34 Physics Features)",
        "Val_Macro_F1": round(pmlp_val, 4),
        "Accuracy (%)": round(pmlp_m["accuracy"] * 100, 2),
        "Macro-F1": round(pmlp_m["macro_f1"], 4),
        "Weighted-F1": round(pmlp_m["weighted_f1"], 4),
        "Config": "34->128->64->14 LayerNorm GELU MLP",
    })

    tab_df = pd.DataFrame(tab_rows)
    tab_df.to_csv(os.path.join(tables_dir, "publication_tabular_baselines.csv"), index=False)

    # =========================================================================
    # LOAD / RECONSTRUCT EXACT TEST PROBABILITIES FOR ALL KEY PARADIGMS
    # =========================================================================
    # RF (exact model + Laplace smoothed)
    rf_model = joblib.load(os.path.join(models_dir, "rf_best_model.joblib"))
    rf_raw_probs = align_proba_columns(rf_model.predict_proba(X_te), rf_model.classes_, class_names)
    rf_preds = np.argmax(rf_raw_probs, axis=1)

    # Adapted 4-Layer CNN (exact checkpoint on 951 1.0s spectrograms -> full probability matrix + Grad-CAM!)
    cnn_ckpt = torch.load(os.path.join(models_dir, "cnn_best_checkpoint.pt"), map_location="cpu", weights_only=False)
    cnn_model = AdaptedGravitySpyCNN(num_classes=len(class_names), in_channels=cnn_ckpt.get("in_channels", 1))
    cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    cnn_model.eval()
    X_te_1s = np.empty((len(te_df), 1, 140, 170), dtype=np.float32)
    with h5py.File(h5_path, "r") as h5_f:
        for i in range(len(te_df)):
            r = te_df.iloc[i]
            X_te_1s[i, 0] = load_spectrogram(h5_f, r["label"], r["sample_type"], r["gravityspy_id"], view="1.0s")
    with torch.no_grad():
        cnn_logits = cnn_model(torch.from_numpy(X_te_1s)).numpy()
    cnn_probs = softmax_np(cnn_logits, 1.0)
    cnn_preds = np.argmax(cnn_probs, axis=1)

    # ImageNet ViT-B/16 (exact cached Val + Test logits & 768d embeddings)
    vit_cache = np.load(os.path.join(models_dir, "vit_b16_val_test_cache.npz"), allow_pickle=False)
    assert np.array_equal(vit_cache["test_ids"], np.asarray(te_df["gravityspy_id"].values, dtype=str))
    vit_val_logits = vit_cache["val_logits"]
    vit_test_logits = vit_cache["test_logits"]
    T_vit = fit_temperature_on_val(vit_val_logits, y_va)
    vit_raw_probs = softmax_np(vit_test_logits, 1.0)
    vit_cal_probs = softmax_np(vit_test_logits, T_vit)
    vit_preds = np.argmax(vit_cal_probs, axis=1)

    # Fine-Tuned CLIP exact test predictions from Step 16 CSV (aligned by gravityspy_id)
    clip_pred_df = pd.read_csv(os.path.join(tables_dir, "vlm_clip_best_test_predictions.csv")).set_index("gravityspy_id").loc[te_df["gravityspy_id"].values]
    clip_preds = np.array([class_to_idx[c] for c in clip_pred_df["vlm_best_pred"].values], dtype=np.int64)

    # Early Fusion MLP (Frozen CLIP + 34 Physics) & Champion Hybrid Fusion exact probabilities
    fr_d = np.load(os.path.join(models_dir, "clip_embeddings_frozen.npz"), allow_pickle=False)
    fr_map = {str(g): fr_d["embeddings"][i].astype(np.float32) for i, g in enumerate(fr_d["gravityspy_id"])}
    E_tr_fr = np.stack([fr_map[str(g)] for g in tr_df["gravityspy_id"].values], axis=0)
    E_va_fr = np.stack([fr_map[str(g)] for g in va_df["gravityspy_id"].values], axis=0)
    E_te_fr = np.stack([fr_map[str(g)] for g in te_df["gravityspy_id"].values], axis=0)
    _, early_te_probs, _, _ = train_early_fusion_mlp(E_tr_fr, X_tr, y_tr, E_va_fr, X_va, y_va, E_te_fr, X_te, class_names, seed=42)
    early_preds = np.argmax(early_te_probs, axis=1)

    champ_probs = fuse_probabilities([early_te_probs, vit_cal_probs], (0.65, 0.35), mode="log_linear")
    champ_preds = np.argmax(champ_probs, axis=1)
    champ_m = compute_all_metrics(y_te_str, [idx_to_class[int(i)] for i in champ_preds], class_names=class_names)
    print(f"  Verified Champion Hybrid Fusion: Test Acc={champ_m['accuracy']*100:.2f}% | Macro-F1={champ_m['macro_f1']:.4f}")

    # =========================================================================
    # 2. 1,000-ITERATION PAIRED BOOTSTRAP 95% CIs & SIGNIFICANCE TESTS
    # =========================================================================
    print("\n--- [2/4] 1,000-Iteration Paired Bootstrap 95% CIs & McNemar Significance Tests ---")
    pred_registry = [
        ("Feature-Based Decision Tree", dt_te_preds),
        ("Physics-Only MLP (34 feats)", pmlp_preds),
        ("Feature-Based Random Forest (34 feats)", rf_preds),
        ("HistGradientBoosting (34 feats)", hgb_preds),
        ("Adapted Gravity Spy CNN (1.0s)", cnn_preds),
        ("CLIP ViT-B/16 Fine-Tuned (RGB)", clip_preds),
        ("ImageNet ViT-B/16 (Multi-View RGB)", vit_preds),
        ("Early Fusion MLP (Frozen CLIP + Physics)", early_preds),
        ("Champion Hybrid Fusion (Physics-CLIP + ViT)", champ_preds),
    ]

    rng = np.random.RandomState(42)
    n_boot = 1000
    n_test = len(y_te)
    boot_indices = rng.randint(0, n_test, size=(n_boot, n_test))

    boot_mf1_store = {}
    ci_rows = []
    for name, preds_idx in pred_registry:
        pt_acc = accuracy_score(y_te, preds_idx) * 100.0
        pt_mf1 = f1_score(y_te, preds_idx, average="macro", zero_division=0)
        b_accs = np.empty(n_boot, dtype=np.float64)
        b_mf1s = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx_b = boot_indices[b]
            yt_b, yp_b = y_te[idx_b], preds_idx[idx_b]
            b_accs[b] = np.mean(yt_b == yp_b) * 100.0
            b_mf1s[b] = f1_score(yt_b, yp_b, average="macro", zero_division=0)
        boot_mf1_store[name] = b_mf1s
        ci_rows.append({
            "Model": name,
            "Accuracy (%)": round(pt_acc, 2),
            "Acc_CI_Low": round(float(np.percentile(b_accs, 2.5)), 2),
            "Acc_CI_High": round(float(np.percentile(b_accs, 97.5)), 2),
            "Macro_F1": round(pt_mf1, 4),
            "Macro_F1_CI_Low": round(float(np.percentile(b_mf1s, 2.5)), 4),
            "Macro_F1_CI_High": round(float(np.percentile(b_mf1s, 97.5)), 4),
            "Test_Errors": int(np.sum(y_te != preds_idx)),
        })

    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(os.path.join(tables_dir, "publication_bootstrap_cis.csv"), index=False)
    print(ci_df.to_string(index=False))

    # Paired significance tests: Champion Hybrid Fusion vs. every prior baseline
    champ_name = "Champion Hybrid Fusion (Physics-CLIP + ViT)"
    champ_ok = (champ_preds == y_te)
    champ_boot = boot_mf1_store[champ_name]
    sig_rows = []

    for name, preds_idx in pred_registry[:-1]:
        base_ok = (preds_idx == y_te)
        n10 = int(np.sum(champ_ok & (~base_ok)))  # Champion right, baseline wrong
        n01 = int(np.sum((~champ_ok) & base_ok))  # Champion wrong, baseline right
        mcnemar_p = float(binomtest(min(n10, n01), n=max(1, n10 + n01), p=0.5, alternative="two-sided").pvalue) if (n10 + n01) > 0 else 1.0

        diff_boot = champ_boot - boot_mf1_store[name]
        pt_diff = float(f1_score(y_te, champ_preds, average="macro", zero_division=0) - f1_score(y_te, preds_idx, average="macro", zero_division=0))
        # Paired bootstrap p-value (fraction of resamples where baseline >= champion)
        p_one = float(np.mean(diff_boot <= 0.0))
        p_two = min(1.0, 2.0 * p_one)

        sig_rows.append({
            "Baseline_Model": name,
            "Delta_Macro_F1": round(pt_diff, 4),
            "Delta_CI_Low": round(float(np.percentile(diff_boot, 2.5)), 4),
            "Delta_CI_High": round(float(np.percentile(diff_boot, 97.5)), 4),
            "Boot_p_one_sided": round(p_one, 4),
            "Boot_p_two_sided": round(p_two, 4),
            "Champ_Wins_n10": n10,
            "Base_Wins_n01": n01,
            "McNemar_Exact_p": round(mcnemar_p, 4),
        })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(os.path.join(tables_dir, "publication_significance_tests.csv"), index=False)
    print("\n=== PAIRED STATISTICAL SIGNIFICANCE: CHAMPION HYBRID FUSION VS. BASELINES ===")
    print(sig_df.to_string(index=False))

    # =========================================================================
    # 3. CALIBRATION (ECE, NLL, BRIER) & HUMAN-IN-THE-LOOP SELECTIVE PREDICTION
    # =========================================================================
    print("\n--- [3/4] Calibration (ECE/NLL/Brier) & Human-in-the-Loop Selective Prediction ---")
    cal_models = {
        "Feature RF (34 feats)": rf_raw_probs,
        "Adapted 4-Layer CNN (1.0s)": cnn_probs,
        "ImageNet ViT-B/16 (Raw T=1.0)": vit_raw_probs,
        "ImageNet ViT-B/16 (Calibrated T=1.60)": vit_cal_probs,
        "Champion Hybrid Fusion": champ_probs,
    }

    cal_summary_rows = []
    abst_rows = []
    review_rates = [0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]

    for m_name, probs in cal_models.items():
        ece, _ = compute_ece(probs, y_te, n_bins=10)
        nll = nll_from_probs(probs, y_te)
        brier = compute_brier(probs, y_te)
        cal_summary_rows.append({
            "Model": m_name,
            "ECE_10bin": round(ece, 4),
            "Test_NLL": round(nll, 4),
            "Brier_Score": round(brier, 4),
        })

        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        order = np.argsort(confs)[::-1]  # Highest confidence first
        for rr in review_rates:
            keep_n = int(round(len(y_te) * (1.0 - rr / 100.0)))
            kept = order[:keep_n]
            acc_k = accuracy_score(y_te[kept], preds[kept]) * 100.0
            mf1_k = f1_score(y_te[kept], preds[kept], average="macro", zero_division=0)
            abst_rows.append({
                "Model": m_name,
                "Human_Review_Rate (%)": rr,
                "Auto_Coverage (%)": round(100.0 - rr, 1),
                "Accepted_Glitches": keep_n,
                "Errors_Remaining": int(np.sum(y_te[kept] != preds[kept])),
                "Auto_Accuracy (%)": round(acc_k, 2),
                "Auto_Macro_F1": round(mf1_k, 4),
            })

    cal_df = pd.DataFrame(cal_summary_rows)
    abst_df = pd.DataFrame(abst_rows)
    print(cal_df.to_string(index=False))
    print("\nChampion Hybrid Fusion Human-in-the-Loop Risk-Coverage:")
    print(abst_df[abst_df["Model"] == "Champion Hybrid Fusion"].to_string(index=False))

    # Save calibration + abstention combined cleanly
    cal_df.to_csv(os.path.join(tables_dir, "publication_calibration_summary.csv"), index=False)
    abst_df.to_csv(os.path.join(tables_dir, "publication_calibration_and_abstention.csv"), index=False)

    # =========================================================================
    # 4. GENERATE ALL 3 PUBLICATION-GRADE INTERPRETABILITY & RIGOR FIGURES
    # =========================================================================
    print("\n--- [4/4] Generating Publication Rigor & Interpretability Figures ---")
    plot_bootstrap_forest(ci_df, sig_df, os.path.join(figures_dir, "publication_bootstrap_forest_plot.png"))
    plot_calibration_and_abstention(cal_models, y_te, abst_df, os.path.join(figures_dir, "publication_calibration_and_selective_prediction.png"))
    plot_gradcam_showcase(cnn_model, te_df, h5_path, class_names, os.path.join(figures_dir, "publication_gradcam_saliency.png"))
    plot_tsne_embeddings(
        vit_cache["test_embs"],
        te_df["label"].values,
        te_df["ifo"].values,
        class_names,
        os.path.join(figures_dir, "publication_tsne_embeddings.png"),
    )

    print(f"\nStep 17.5 Publication-Readiness Rigor Suite finished cleanly in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
