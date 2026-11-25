"""
Module: failure_analysis.py
Purpose: Systematic Error & Visual Failure Analysis (Step 14) comparing Random Forest vs Adapted CNN.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Analyzes:
1. Agreement & disagreement matrix between our 34-Feature Random Forest and Adapted 4-Layer CNN on Test (N=951).
2. Look-alike morphological confusion pairs (Blip vs Repeating_Blips, Blip vs Tomte, Low_Freq_Burst vs Lines).
3. Cross-detector failure signatures (where H1 -> L1 and L1 -> H1 errors concentrate).
4. Publication-grade spectrogram grids displaying true vs predicted labels and softmax confidences.
"""

import os
import sys
import json
import h5py
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import load_spectrogram
from src.models.tree_models import ALL_FEATURE_COLS


def extract_misclassified_samples(y_true, y_pred, metadata_df, confidences=None):
    """
    Returns a DataFrame containing only misclassified samples with true label, predicted label,
    and optional confidence score.
    """
    out = metadata_df.copy().reset_index(drop=True)
    out["true_label"] = np.asarray(y_true)
    out["pred_label"] = np.asarray(y_pred)
    if confidences is not None:
        out["confidence"] = np.round(np.asarray(confidences, dtype=float), 4)
    mis = out[out["true_label"] != out["pred_label"]].copy().reset_index(drop=True)
    return mis


def plot_failure_spectrograms(misclassified_df, h5_path, title="Misclassified Glitches", num_examples=9, save_path=None):
    """
    Plots a 3x3 visual grid of misclassified 1.0s spectrograms showing True label,
    RF prediction, CNN prediction, detector site, and glitch ID.
    """
    sub = misclassified_df.head(num_examples).reset_index(drop=True)
    n = len(sub)
    if n == 0:
        print("No misclassified samples to plot.")
        return

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    with h5py.File(h5_path, "r") as h5_f:
        for i in range(len(axes)):
            ax = axes[i]
            if i >= n:
                ax.axis("off")
                continue

            row = sub.iloc[i]
            img = load_spectrogram(h5_f, row["true_label"], row["sample_type"], row["gravityspy_id"], view="1.0s")
            ax.imshow(img, cmap="viridis", origin="lower", aspect="auto")

            rf_tag = f"RF: {row['rf_pred']} ({row['rf_conf']:.2f})" if "rf_pred" in row else ""
            cnn_tag = f"CNN: {row['cnn_pred']} ({row['cnn_conf']:.2f})" if "cnn_pred" in row else f"Pred: {row.get('pred_label', '')}"
            header = f"True: {row['true_label']}  [{row['ifo']} | {row['gravityspy_id']}]"
            sub_lbl = f"{rf_tag}  |  {cnn_tag}" if rf_tag else cnn_tag

            ax.set_title(f"{header}\n{sub_lbl}", fontsize=9.2, fontweight="bold", pad=6)
            ax.set_xlabel("Time Bins (1.0s)", fontsize=8.5)
            ax.set_ylabel("Freq Bins", fontsize=8.5)

    plt.suptitle(title, fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=250)
        print(f"Failure spectrogram grid saved to: {save_path}")
    plt.close(fig)


def plot_disagreement_showcase(cnn_wins_df, rf_wins_df, h5_path, save_path):
    """
    Plots a 2x3 comparative showcase:
    - Top row: Glitches where Adapted CNN was CORRECT and Feature RF was WRONG.
    - Bottom row: Glitches where Feature RF was CORRECT and Adapted CNN was WRONG.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    with h5py.File(h5_path, "r") as h5_f:
        # Row 0: CNN beats RF
        for col in range(3):
            ax = axes[0, col]
            if col >= len(cnn_wins_df):
                ax.axis("off")
                continue
            row = cnn_wins_df.iloc[col]
            img = load_spectrogram(h5_f, row["true_label"], row["sample_type"], row["gravityspy_id"], view="1.0s")
            ax.imshow(img, cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(
                f"[CNN Wins] True: {row['true_label']} ({row['ifo']})\n"
                f"CNN: {row['cnn_pred']} ({row['cnn_conf']:.2f})  |  RF: {row['rf_pred']} ({row['rf_conf']:.2f})",
                fontsize=9.2, fontweight="bold", color="#1b5e20"
            )
            ax.set_xlabel("Time Bins (1.0s)", fontsize=8.5)
            ax.set_ylabel("Freq Bins", fontsize=8.5)

        # Row 1: RF beats CNN
        for col in range(3):
            ax = axes[1, col]
            if col >= len(rf_wins_df):
                ax.axis("off")
                continue
            row = rf_wins_df.iloc[col]
            img = load_spectrogram(h5_f, row["true_label"], row["sample_type"], row["gravityspy_id"], view="1.0s")
            ax.imshow(img, cmap="viridis", origin="lower", aspect="auto")
            ax.set_title(
                f"[RF Wins] True: {row['true_label']} ({row['ifo']})\n"
                f"RF: {row['rf_pred']} ({row['rf_conf']:.2f})  |  CNN: {row['cnn_pred']} ({row['cnn_conf']:.2f})",
                fontsize=9.2, fontweight="bold", color="#0d47a1"
            )
            ax.set_xlabel("Time Bins (1.0s)", fontsize=8.5)
            ax.set_ylabel("Freq Bins", fontsize=8.5)

    plt.suptitle("Complementary Strengths: Where CNN Beats RF (Top) vs. Where RF Beats CNN (Bottom)", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=250)
    plt.close(fig)
    print(f"Model disagreement showcase saved to: {save_path}")


def run_full_failure_analysis():
    features_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    rf_model_path = os.path.join(project_root, "outputs", "models", "rf_best_model.joblib")
    cnn_preds_csv = os.path.join(project_root, "outputs", "tables", "cnn_test_predictions.csv")
    cross_preds_csv = os.path.join(project_root, "outputs", "tables", "cross_detector_predictions_seed42.csv")
    h5_path = os.path.join(project_root, "trainingsetv1d1.h5")
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")

    print("=" * 72)
    print(" STEP 14: VISUAL FAILURE ANALYSIS & MODEL DISAGREEMENT STUDY")
    print("=" * 72)

    # 1. Load test partition and compute RF predictions + confidences
    feat_df = pd.read_csv(features_csv)
    with open(splits_json, "r", encoding="utf-8") as f:
        split_registry = json.load(f)
    test_ids = set(split_registry["default_split"]["test_ids"])
    test_df = feat_df[feat_df["gravityspy_id"].isin(test_ids)].copy().reset_index(drop=True)

    rf_model = joblib.load(rf_model_path)
    X_test_rf = test_df[ALL_FEATURE_COLS].values.astype(np.float32)
    rf_preds = rf_model.predict(X_test_rf)
    rf_confs = np.max(rf_model.predict_proba(X_test_rf), axis=1)

    rf_res_df = pd.DataFrame({
        "gravityspy_id": test_df["gravityspy_id"].values,
        "rf_pred": rf_preds,
        "rf_conf": np.round(rf_confs, 4),
        "rf_correct": (test_df["label"].values == rf_preds)
    })

    # 2. Merge with CNN test predictions
    cnn_df = pd.read_csv(cnn_preds_csv).rename(columns={"cnn_confidence": "cnn_conf"})
    merged = cnn_df.merge(rf_res_df, on="gravityspy_id", how="inner")

    # 3. Compute 2x2 agreement breakdown
    both_correct = merged[merged["rf_correct"] & merged["cnn_correct"]]
    rf_only = merged[merged["rf_correct"] & (~merged["cnn_correct"])].sort_values(by="cnn_conf", ascending=False)
    cnn_only = merged[(~merged["rf_correct"]) & merged["cnn_correct"]].sort_values(by="rf_conf", ascending=False)
    both_wrong = merged[(~merged["rf_correct"]) & (~merged["cnn_correct"])].sort_values(by="rf_conf", ascending=False)

    summary_df = pd.DataFrame([
        {"Category": "Both RF & CNN Correct", "Count": len(both_correct), "Percentage (%)": round(len(both_correct) / len(merged) * 100, 2)},
        {"Category": "RF Correct, CNN Wrong (Physics Features Win)", "Count": len(rf_only), "Percentage (%)": round(len(rf_only) / len(merged) * 100, 2)},
        {"Category": "CNN Correct, RF Wrong (2D Vision Wins)", "Count": len(cnn_only), "Percentage (%)": round(len(cnn_only) / len(merged) * 100, 2)},
        {"Category": "Both RF & CNN Wrong (Hard / Ambiguous Glitches)", "Count": len(both_wrong), "Percentage (%)": round(len(both_wrong) / len(merged) * 100, 2)},
    ])
    summary_df.to_csv(os.path.join(tables_dir, "model_disagreement_summary.csv"), index=False)
    print("\n=== RF vs. CNN AGREEMENT BREAKDOWN (Unseen Test Set, N=951) ===")
    print(summary_df.to_string(index=False))

    # 4. Save all failure cases where at least one model erred
    any_error = merged[(~merged["rf_correct"]) | (~merged["cnn_correct"])].copy()
    any_error.to_csv(os.path.join(tables_dir, "failure_cases_detailed.csv"), index=False)

    # Top confusion pairs for both models
    print("\n=== TOP CONFUSION PAIRS (BOTH MODELS WRONG) ===")
    if len(both_wrong) > 0:
        pair_counts = both_wrong.groupby(["true_label", "rf_pred", "cnn_pred"]).size().reset_index(name="count").sort_values(by="count", ascending=False)
        print(pair_counts.head(10).to_string(index=False))

    # 5. Select 9 diverse hard/look-alike failures (prioritizing Both Wrong, then look-alikes)
    priority_df = pd.concat([both_wrong, cnn_only, rf_only], ignore_index=True).drop_duplicates(subset=["gravityspy_id"])
    # Pick diverse true classes so the 3x3 grid spans look-alike morphologies
    selected_rows = []
    seen_classes = {}
    for _, row in priority_df.iterrows():
        cls = row["true_label"]
        if seen_classes.get(cls, 0) < 2:
            selected_rows.append(row)
            seen_classes[cls] = seen_classes.get(cls, 0) + 1
        if len(selected_rows) == 9:
            break
    grid_df = pd.DataFrame(selected_rows)

    plot_failure_spectrograms(
        grid_df, h5_path,
        title="Step 14: Visual Failure Analysis — Hard & Look-Alike Misclassifications (1.0s Spectrograms)",
        num_examples=9,
        save_path=os.path.join(figures_dir, "failure_cases_spectrograms.png")
    )

    # 6. Plot complementary strengths showcase (CNN Wins vs RF Wins)
    plot_disagreement_showcase(
        cnn_only.head(3).reset_index(drop=True),
        rf_only.head(3).reset_index(drop=True),
        h5_path,
        save_path=os.path.join(figures_dir, "rf_vs_cnn_disagreements.png")
    )

    # 7. Cross-detector failure signature summary (if Step 13 ran)
    if os.path.exists(cross_preds_csv):
        cd_df = pd.read_csv(cross_preds_csv)
        cd_errs = cd_df[~cd_df["correct"]]
        print("\n=== TOP CROSS-DETECTOR TRANSFER ERRORS (SEED 42) ===")
        for direction in ["H1->L1", "L1->H1"]:
            for mod in ["RF", "CNN"]:
                sub = cd_errs[(cd_errs["direction"] == direction) & (cd_errs["model"] == mod)]
                top_pairs = sub.groupby(["true_label", "pred_label"]).size().reset_index(name="count").sort_values(by="count", ascending=False).head(4)
                pairs_str = ", ".join(f"{r['true_label']}->{r['pred_label']} ({r['count']})" for _, r in top_pairs.iterrows())
                print(f"  [{mod:3s} | {direction}] Total Errors={len(sub):3d} | Top Pairs: {pairs_str}")

    print("\nStep 14 failure analysis completed successfully!")


if __name__ == "__main__":
    run_full_failure_analysis()
