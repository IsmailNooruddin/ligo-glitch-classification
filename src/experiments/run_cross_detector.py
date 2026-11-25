"""
Module: run_cross_detector.py
Purpose: Option B — Bidirectional Cross-Detector Transfer Testing (H1 <-> L1) across 5 random seeds.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Directly fulfills the instructor's main extension mandate:
- Evaluates transfer in BOTH directions on the 14 shared classes:
    1. H1 -> H1 (Hanford within-detector control)
    2. H1 -> L1 (Cross-detector transfer: trained on Hanford, tested on unseen Livingston)
    3. L1 -> L1 (Livingston within-detector control)
    4. L1 -> H1 (Cross-detector transfer: trained on Livingston, tested on unseen Hanford)
- Runs across 5 random seeds ([42, 43, 44, 45, 46]) with zero gravityspy_id leakage.
- Compares BOTH Feature-Based Random Forest (34 features) AND Adapted Gravity Spy CNN (1.0s spectrograms).
- Reports Macro-F1 (mean ± std dev), Accuracy (mean ± std dev), Transfer Drop (Delta), and Per-Class F1 breakdowns.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.split_utils import create_stratified_glitch_split
from src.data.dataset import get_class_mapping
from src.models.tree_models import ALL_FEATURE_COLS
from src.evaluation.metrics import compute_all_metrics
from src.experiments.run_cnn import (
    preload_spectrogram_dict,
    build_tensor_arrays,
    train_cnn_model,
    evaluate_cnn_model
)


SEEDS = [42, 43, 44, 45, 46]
DIRECTIONS = ["H1->H1", "H1->L1", "L1->L1", "L1->H1"]


def plot_cross_detector_comparison(summary_df, save_path):
    """
    Plots grouped bar chart with error bars comparing Random Forest vs Adapted CNN
    across H1->H1, H1->L1, L1->L1, and L1->H1 (Mean Macro-F1 ± Std across 5 seeds).
    """
    directions = DIRECTIONS
    x = np.arange(len(directions))
    width = 0.35

    rf_sub = summary_df[summary_df["Model"] == "Feature Random Forest"].set_index("Direction").loc[directions]
    cnn_sub = summary_df[summary_df["Model"] == "Adapted Gravity Spy CNN"].set_index("Direction").loc[directions]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_rf = ax.bar(
        x - width / 2, rf_sub["Macro_F1_Mean"], width,
        yerr=rf_sub["Macro_F1_Std"], capsize=5,
        label="Feature Random Forest (34 Features)", color="#2ca02c", alpha=0.88, edgecolor="black"
    )
    bars_cnn = ax.bar(
        x + width / 2, cnn_sub["Macro_F1_Mean"], width,
        yerr=cnn_sub["Macro_F1_Std"], capsize=5,
        label="Adapted Gravity Spy CNN (1.0s Spectrogram)", color="#1f77b4", alpha=0.88, edgecolor="black"
    )

    for bars, stds in [(bars_rf, rf_sub["Macro_F1_Std"]), (bars_cnn, cnn_sub["Macro_F1_Std"])]:
        for bar, sd in zip(bars, stds):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + sd + 0.012,
                f"{h:.3f}\n±{sd:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels([
        "H1 → H1\n(Within Hanford)",
        "H1 → L1\n(Cross-Site Transfer)",
        "L1 → L1\n(Within Livingston)",
        "L1 → H1\n(Cross-Site Transfer)"
    ], fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Test Macro-F1 Score (5-Seed Mean ± Std)", fontsize=11, fontweight="bold")
    ax.set_title("Option B: Bidirectional Cross-Detector Transfer (Hanford H1 ↔ Livingston L1)\nFeature Random Forest vs. Adapted Gravity Spy CNN (14 Shared Classes, 5 Seeds)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Cross-detector comparison figure saved to: {save_path}")


def plot_per_class_transfer_heatmap(per_class_df, save_path):
    """
    Plots side-by-side heatmaps showing mean per-class F1 across all 4 directions
    for Feature Random Forest vs Adapted Gravity Spy CNN.
    """
    import seaborn as sns
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

    for ax, model_name, title_tag in [
        (axes[0], "Feature Random Forest", "Feature Random Forest (34 Features)"),
        (axes[1], "Adapted Gravity Spy CNN", "Adapted Gravity Spy CNN (1.0s)")
    ]:
        sub = per_class_df[per_class_df["Model"] == model_name].set_index("Class")[DIRECTIONS]
        sns.heatmap(
            sub, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.0, vmax=1.0,
            cbar_kws={"label": "Mean F1 (5 Seeds)"}, ax=ax
        )
        ax.set_title(title_tag, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Train → Test Detector", fontsize=10, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Shared Glitch Class", fontsize=10, fontweight="bold")

    plt.suptitle("Per-Class Cross-Detector F1 Transfer Profile (Mean Across 5 Seeds)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Per-class cross-detector heatmap saved to: {save_path}")


def main(cnn_epochs=14, cnn_patience=4):
    features_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    meta_csv = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    h5_path = os.path.join(project_root, "trainingsetv1d1.h5")
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 75)
    print(" STEP 13: OPTION B — BIDIRECTIONAL CROSS-DETECTOR TRANSFER (5 SEEDS)")
    print("=" * 75)

    feat_df = pd.read_csv(features_csv)
    meta_df = pd.read_csv(meta_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)

    class_names = split_registry["classes"]
    class_to_idx, _ = get_class_mapping(class_names)

    # Split metadata by detector site
    h1_meta = meta_df[meta_df["ifo"] == "H1"].copy().reset_index(drop=True)
    l1_meta = meta_df[meta_df["ifo"] == "L1"].copy().reset_index(drop=True)
    print(f"Detector pools (14 shared classes): H1={len(h1_meta)} glitches | L1={len(l1_meta)} glitches")

    # Index engineered features by gravityspy_id for fast lookup
    feat_indexed = feat_df.set_index("gravityspy_id")

    # Preload all 6,340 1.0s spectrograms once into RAM
    spec_dict = preload_spectrogram_dict(h5_path, meta_df, view="1.0s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_seed_records = []
    per_class_accum = {
        (m, d): {c: [] for c in class_names}
        for m in ["Feature Random Forest", "Adapted Gravity Spy CNN"]
        for d in DIRECTIONS
    }
    seed42_cross_preds = []

    for seed in SEEDS:
        print("\n" + "=" * 75)
        print(f" RUNNING SEED {seed} / {SEEDS}")
        print("=" * 75)

        # Leak-free stratified grouped splits for H1 and L1 independently
        h1_split = create_stratified_glitch_split(h1_meta, test_size=0.15, val_size=0.15, random_state=seed)
        l1_split = create_stratified_glitch_split(l1_meta, test_size=0.15, val_size=0.15, random_state=seed)

        h1_tr = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_split["train_ids"]))].copy().reset_index(drop=True)
        h1_va = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_split["val_ids"]))].copy().reset_index(drop=True)
        h1_te = h1_meta[h1_meta["gravityspy_id"].isin(set(h1_split["test_ids"]))].copy().reset_index(drop=True)

        l1_tr = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_split["train_ids"]))].copy().reset_index(drop=True)
        l1_va = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_split["val_ids"]))].copy().reset_index(drop=True)
        l1_te = l1_meta[l1_meta["gravityspy_id"].isin(set(l1_split["test_ids"]))].copy().reset_index(drop=True)

        # Verify zero leakage
        assert set(h1_tr["gravityspy_id"]).isdisjoint(set(h1_te["gravityspy_id"]))
        assert set(l1_tr["gravityspy_id"]).isdisjoint(set(l1_te["gravityspy_id"]))

        # Prepare RF feature arrays
        X_h1_tr_rf = feat_indexed.loc[h1_tr["gravityspy_id"], ALL_FEATURE_COLS].values.astype(np.float32)
        y_h1_tr_str = h1_tr["label"].values
        X_h1_te_rf = feat_indexed.loc[h1_te["gravityspy_id"], ALL_FEATURE_COLS].values.astype(np.float32)
        y_h1_te_str = h1_te["label"].values

        X_l1_tr_rf = feat_indexed.loc[l1_tr["gravityspy_id"], ALL_FEATURE_COLS].values.astype(np.float32)
        y_l1_tr_str = l1_tr["label"].values
        X_l1_te_rf = feat_indexed.loc[l1_te["gravityspy_id"], ALL_FEATURE_COLS].values.astype(np.float32)
        y_l1_te_str = l1_te["label"].values

        # ------------------------------------------------------------------
        # A. Feature Random Forest (Trained on H1, then Trained on L1)
        # ------------------------------------------------------------------
        rf_h1 = RandomForestClassifier(n_estimators=50, max_depth=None, max_features="sqrt", random_state=seed, n_jobs=-1)
        rf_h1.fit(X_h1_tr_rf, y_h1_tr_str)

        rf_l1 = RandomForestClassifier(n_estimators=50, max_depth=None, max_features="sqrt", random_state=seed, n_jobs=-1)
        rf_l1.fit(X_l1_tr_rf, y_l1_tr_str)

        rf_evals = [
            ("H1->H1", rf_h1.predict(X_h1_te_rf), y_h1_te_str, h1_te),
            ("H1->L1", rf_h1.predict(X_l1_te_rf), y_l1_te_str, l1_te),
            ("L1->L1", rf_l1.predict(X_l1_te_rf), y_l1_te_str, l1_te),
            ("L1->H1", rf_l1.predict(X_h1_te_rf), y_h1_te_str, h1_te),
        ]

        for direction, y_pred, y_true, eval_meta in rf_evals:
            m = compute_all_metrics(y_true, y_pred, class_names=class_names)
            per_seed_records.append({
                "Model": "Feature Random Forest",
                "Direction": direction,
                "Seed": seed,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Macro_F1": round(m["macro_f1"], 4),
                "Weighted_F1": round(m["weighted_f1"], 4)
            })
            for cls_name, f1_val in zip(m["per_class"]["Class"], m["per_class"]["F1-Score"]):
                per_class_accum[("Feature Random Forest", direction)][cls_name].append(float(f1_val))
            print(f"  [RF  | Seed {seed}] {direction}: Acc={m['accuracy']*100:5.2f}% | Macro-F1={m['macro_f1']:.4f}")

            if seed == 42 and direction in ("H1->L1", "L1->H1"):
                for gid, ifo, stype, yt, yp in zip(
                    eval_meta["gravityspy_id"], eval_meta["ifo"], eval_meta["sample_type"], y_true, y_pred
                ):
                    seed42_cross_preds.append({
                        "model": "RF", "direction": direction, "gravityspy_id": gid,
                        "ifo": ifo, "sample_type": stype, "true_label": yt, "pred_label": yp, "correct": yt == yp
                    })

        # ------------------------------------------------------------------
        # B. Adapted Gravity Spy CNN (Trained on H1, then Trained on L1)
        # ------------------------------------------------------------------
        X_h1_tr_c, y_h1_tr_c = build_tensor_arrays(h1_tr, spec_dict, class_to_idx)
        X_h1_va_c, y_h1_va_c = build_tensor_arrays(h1_va, spec_dict, class_to_idx)
        X_h1_te_c, y_h1_te_c = build_tensor_arrays(h1_te, spec_dict, class_to_idx)

        X_l1_tr_c, y_l1_tr_c = build_tensor_arrays(l1_tr, spec_dict, class_to_idx)
        X_l1_va_c, y_l1_va_c = build_tensor_arrays(l1_va, spec_dict, class_to_idx)
        X_l1_te_c, y_l1_te_c = build_tensor_arrays(l1_te, spec_dict, class_to_idx)

        print(f"  [CNN | Seed {seed}] Training H1 CNN (Train={len(X_h1_tr_c)}, Val={len(X_h1_va_c)})...")
        cnn_h1, _, val_f1_h1, t_h1 = train_cnn_model(
            X_h1_tr_c, y_h1_tr_c, X_h1_va_c, y_h1_va_c, class_names,
            in_channels=1, seed=seed, epochs=cnn_epochs, batch_size=64,
            lr=1.5e-3, patience=cnn_patience, device=device, verbose=False
        )
        print(f"    -> H1 CNN finished in {t_h1:.1f}s (Best Val Macro-F1: {val_f1_h1:.4f})")

        print(f"  [CNN | Seed {seed}] Training L1 CNN (Train={len(X_l1_tr_c)}, Val={len(X_l1_va_c)})...")
        cnn_l1, _, val_f1_l1, t_l1 = train_cnn_model(
            X_l1_tr_c, y_l1_tr_c, X_l1_va_c, y_l1_va_c, class_names,
            in_channels=1, seed=seed, epochs=cnn_epochs, batch_size=64,
            lr=1.5e-3, patience=cnn_patience, device=device, verbose=False
        )
        print(f"    -> L1 CNN finished in {t_l1:.1f}s (Best Val Macro-F1: {val_f1_l1:.4f})")

        cnn_evals = [
            ("H1->H1", *evaluate_cnn_model(cnn_h1, X_h1_te_c, y_h1_te_c, class_names, device), h1_te),
            ("H1->L1", *evaluate_cnn_model(cnn_h1, X_l1_te_c, y_l1_te_c, class_names, device), l1_te),
            ("L1->L1", *evaluate_cnn_model(cnn_l1, X_l1_te_c, y_l1_te_c, class_names, device), l1_te),
            ("L1->H1", *evaluate_cnn_model(cnn_l1, X_h1_te_c, y_h1_te_c, class_names, device), h1_te),
        ]

        for direction, y_pred, probs, m, eval_meta in cnn_evals:
            per_seed_records.append({
                "Model": "Adapted Gravity Spy CNN",
                "Direction": direction,
                "Seed": seed,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Macro_F1": round(m["macro_f1"], 4),
                "Weighted_F1": round(m["weighted_f1"], 4)
            })
            for cls_name, f1_val in zip(m["per_class"]["Class"], m["per_class"]["F1-Score"]):
                per_class_accum[("Adapted Gravity Spy CNN", direction)][cls_name].append(float(f1_val))
            print(f"  [CNN | Seed {seed}] {direction}: Acc={m['accuracy']*100:5.2f}% | Macro-F1={m['macro_f1']:.4f}")

            if seed == 42 and direction in ("H1->L1", "L1->H1"):
                y_true = eval_meta["label"].values
                confs = np.max(probs, axis=1)
                for gid, ifo, stype, yt, yp, cf in zip(
                    eval_meta["gravityspy_id"], eval_meta["ifo"], eval_meta["sample_type"], y_true, y_pred, confs
                ):
                    seed42_cross_preds.append({
                        "model": "CNN", "direction": direction, "gravityspy_id": gid,
                        "ifo": ifo, "sample_type": stype, "true_label": yt, "pred_label": yp,
                        "confidence": round(float(cf), 4), "correct": yt == yp
                    })

    # ----------------------------------------------------------------------
    # Aggregate Mean ± Std across all 5 seeds
    # ----------------------------------------------------------------------
    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_csv = os.path.join(tables_dir, "cross_detector_per_seed.csv")
    per_seed_df.to_csv(per_seed_csv, index=False)

    summary_rows = []
    for model_name in ["Feature Random Forest", "Adapted Gravity Spy CNN"]:
        sub_m = per_seed_df[per_seed_df["Model"] == model_name]
        # Compute mean baseline within-site values for transfer drop calculation
        h1_within = sub_m[sub_m["Direction"] == "H1->H1"]["Macro_F1"].mean()
        l1_within = sub_m[sub_m["Direction"] == "L1->L1"]["Macro_F1"].mean()

        for direction in DIRECTIONS:
            sub_d = sub_m[sub_m["Direction"] == direction]
            mf1_mean = float(sub_d["Macro_F1"].mean())
            mf1_std = float(sub_d["Macro_F1"].std(ddof=0))
            acc_mean = float(sub_d["Accuracy (%)"].mean())
            acc_std = float(sub_d["Accuracy (%)"].std(ddof=0))

            if direction == "H1->L1":
                drop = mf1_mean - l1_within  # Compared to native L1 model tested on L1
            elif direction == "L1->H1":
                drop = mf1_mean - h1_within  # Compared to native H1 model tested on H1
            else:
                drop = 0.0

            summary_rows.append({
                "Model": model_name,
                "Direction": direction,
                "Accuracy_Mean (%)": round(acc_mean, 2),
                "Accuracy_Std (%)": round(acc_std, 2),
                "Macro_F1_Mean": round(mf1_mean, 4),
                "Macro_F1_Std": round(mf1_std, 4),
                "Transfer_Gap_vs_Target": round(drop, 4),
                "Num_Seeds": len(SEEDS)
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(tables_dir, "cross_detector_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Per-class mean F1 table
    pc_rows = []
    for model_name in ["Feature Random Forest", "Adapted Gravity Spy CNN"]:
        for cls_name in class_names:
            row = {"Model": model_name, "Class": cls_name}
            for direction in DIRECTIONS:
                vals = per_class_accum[(model_name, direction)][cls_name]
                row[direction] = round(float(np.mean(vals)), 4)
                row[f"{direction}_std"] = round(float(np.std(vals)), 4)
            pc_rows.append(row)

    per_class_cd_df = pd.DataFrame(pc_rows)
    pc_cd_csv = os.path.join(tables_dir, "cross_detector_per_class_f1.csv")
    per_class_cd_df.to_csv(pc_cd_csv, index=False)

    # Save Seed 42 cross-detector predictions for Step 14 failure inspection
    pd.DataFrame(seed42_cross_preds).to_csv(os.path.join(tables_dir, "cross_detector_predictions_seed42.csv"), index=False)

    print("\n" + "=" * 85)
    print(" STEP 13 FINAL SUMMARY: BIDIRECTIONAL CROSS-DETECTOR TRANSFER (5 SEEDS)")
    print("=" * 85)
    print(summary_df.to_string(index=False))

    # Generate plots
    plot_cross_detector_comparison(summary_df, os.path.join(figures_dir, "cross_detector_transfer_comparison.png"))
    plot_per_class_transfer_heatmap(per_class_cd_df, os.path.join(figures_dir, "cross_detector_per_class_heatmap.png"))

    print("\nStep 13 completed successfully!")


if __name__ == "__main__":
    main()
