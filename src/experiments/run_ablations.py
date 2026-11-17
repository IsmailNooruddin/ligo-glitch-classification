"""
Module: run_ablations.py
Purpose: Feature importance calculation (MDI + Permutation) and systematic feature-group ablation experiments.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Key Research Questions Addressed:
1. Which physical features provide the strongest discriminative power for distinguishing gravitational-wave glitches?
2. How does model performance degrade when specific physical dimensions (temporal, spectral, energy, morphology) are ablated?
3. Can the 14 tabular Omicron metadata features substitute for the 20 spectrogram-derived features, or are both modalities necessary?
4. How do feature ablations impact individual glitch classes (e.g., does removing frequency features destroy Violin Mode and Power Line detection)?
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.tree_models import (
    ALL_FEATURE_COLS,
    SPECTROGRAM_FEATURE_COLS,
    METADATA_FEATURE_COLS,
    prepare_feature_matrices
)
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


# ==============================================================================
# Define Scientific Feature Groups
# ==============================================================================

TEMPORAL_FEATURES = [
    'meta_duration',
    'meta_log_duration',
    'spec_peak_time_bin',
    'spec_temporal_centroid',
    'spec_temporal_spread',
    'spec_active_duration_bins'
]

FREQUENCY_FEATURES = [
    'meta_peak_frequency',
    'meta_log_peak_frequency',
    'meta_central_freq',
    'meta_log_central_freq',
    'meta_bandwidth',
    'meta_log_bandwidth',
    'meta_bandwidth_ratio',
    'spec_peak_freq_bin',
    'spec_spectral_centroid',
    'spec_spectral_spread',
    'spec_spectral_skewness',
    'spec_low_band_ratio',
    'spec_mid_band_ratio',
    'spec_high_band_ratio'
]

ENERGY_FEATURES = [
    'meta_amplitude',
    'meta_log_amplitude',
    'meta_snr',
    'meta_log_snr',
    'spec_peak_energy',
    'spec_mean_energy',
    'spec_std_energy',
    'spec_energy_contrast',
    'spec_energy_concentration'
]

MORPHOLOGY_FEATURES = [
    'spec_active_pixel_ratio',
    'spec_horizontal_persistence',
    'spec_vertical_persistence',
    'spec_aspect_ratio',
    'meta_q_factor_approx'
]


def compute_feature_importances(rf_model, X_test, y_test, feature_names, class_names, tables_dir, figures_dir):
    """
    Computes both Mean Decrease in Impurity (MDI) and Permutation Feature Importance.
    Saves a tabular ranking and a high-resolution comparison figure.
    """
    print("\n" + "=" * 70)
    print(" [1/3] COMPUTING FEATURE IMPORTANCES (MDI & PERMUTATION)")
    print("=" * 70)

    # 1. Gini Importance (MDI) directly from trained forest
    mdi_importances = rf_model.feature_importances_

    # 2. Permutation Feature Importance on unseen test set (scoring = Macro-F1)
    print("Computing Permutation Importance on test set (10 repeats, metric = Macro-F1)...")
    t0 = time.time()
    perm_result = permutation_importance(
        rf_model,
        X_test,
        y_test,
        scoring='f1_macro',
        n_repeats=10,
        random_state=42,
        n_jobs=1
    )
    perm_duration = time.time() - t0
    print(f"Permutation importance finished in {perm_duration:.2f} seconds.")

    # 3. Compile into DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Group': [
            'Temporal' if f in TEMPORAL_FEATURES else
            'Frequency' if f in FREQUENCY_FEATURES else
            'Energy/SNR' if f in ENERGY_FEATURES else
            'Morphology' for f in feature_names
        ],
        'Modality': ['Spectrogram' if f.startswith('spec_') else 'Omicron Metadata' for f in feature_names],
        'Permutation_Mean': perm_result.importances_mean,
        'Permutation_Std': perm_result.importances_std,
        'MDI_Gini': mdi_importances
    }).sort_values(by='Permutation_Mean', ascending=False).reset_index(drop=True)

    # Save to CSV
    csv_path = os.path.join(tables_dir, "feature_importances.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Feature importance table saved to: {csv_path}")

    print("\nTop 10 Most Critical Features (by Test Set Permutation Macro-F1 Drop):")
    display_cols = ['Feature', 'Group', 'Modality', 'Permutation_Mean', 'Permutation_Std', 'MDI_Gini']
    print(importance_df[display_cols].head(10).to_string(index=False))

    # 4. Generate Publication-Quality Dual-Panel Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=False)

    # Panel A: Permutation Importance (Top 15)
    top_perm = importance_df.head(15).iloc[::-1]  # Reverse for horizontal bar chart
    axes[0].barh(
        top_perm['Feature'],
        top_perm['Permutation_Mean'],
        xerr=top_perm['Permutation_Std'],
        color='#1f77b4',
        alpha=0.85,
        capsize=4,
        edgecolor='black'
    )
    axes[0].set_title("(A) Permutation Importance (Test Set Macro-F1 Drop)", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Mean Decrease in Macro-F1 Score", fontsize=10, fontweight='bold')
    axes[0].grid(axis='x', linestyle='--', alpha=0.6)

    # Panel B: MDI Gini Importance (Top 15)
    top_mdi = importance_df.sort_values(by='MDI_Gini', ascending=False).head(15).iloc[::-1]
    axes[1].barh(
        top_mdi['Feature'],
        top_mdi['MDI_Gini'],
        color='#2ca02c',
        alpha=0.85,
        edgecolor='black'
    )
    axes[1].set_title("(B) Mean Decrease in Impurity (Gini Importance)", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("MDI Score (Fraction of Total Gini Impurity Reduction)", fontsize=10, fontweight='bold')
    axes[1].grid(axis='x', linestyle='--', alpha=0.6)

    plt.suptitle("Feature Importance Analysis for LIGO Glitch Classification (Random Forest)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "feature_importances.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Feature importance figure saved to: {fig_path}")

    return importance_df


def run_ablation_experiments(train_df, val_df, test_df, class_names, tables_dir, figures_dir):
    """
    Performs systematic feature ablations by training the Random Forest on reduced feature subsets.
    Evaluates impact on overall Macro-F1, Accuracy, and per-class performance.
    """
    print("\n" + "=" * 70)
    print(" [2/3] RUNNING SYSTEMATIC FEATURE ABLATION EXPERIMENTS")
    print("=" * 70)

    # Define ablation scenarios
    ablations = [
        {
            "name": "Full Feature Set (Reference)",
            "features": ALL_FEATURE_COLS,
            "description": "All 34 features (20 spectrogram + 14 Omicron metadata)"
        },
        {
            "name": "Drop Temporal Group",
            "features": [f for f in ALL_FEATURE_COLS if f not in TEMPORAL_FEATURES],
            "description": "Ablates duration, peak time, temporal centroid, active duration bins (28 features remaining)"
        },
        {
            "name": "Drop Frequency Group",
            "features": [f for f in ALL_FEATURE_COLS if f not in FREQUENCY_FEATURES],
            "description": "Ablates peak/central frequency, bandwidth, spectral centroid/spread/skewness, sub-bands (20 features remaining)"
        },
        {
            "name": "Drop Energy / SNR Group",
            "features": [f for f in ALL_FEATURE_COLS if f not in ENERGY_FEATURES],
            "description": "Ablates SNR, amplitude, peak/mean energy, energy contrast, energy concentration (25 features remaining)"
        },
        {
            "name": "Drop Morphology Group",
            "features": [f for f in ALL_FEATURE_COLS if f not in MORPHOLOGY_FEATURES],
            "description": "Ablates aspect ratio, persistence, active pixel ratio, Q-factor approx (29 features remaining)"
        },
        {
            "name": "Modality: Spectrogram Only",
            "features": SPECTROGRAM_FEATURE_COLS,
            "description": "Drops all 14 Omicron trigger metadata features (20 spectrogram features remaining)"
        },
        {
            "name": "Modality: Metadata Only",
            "features": METADATA_FEATURE_COLS,
            "description": "Drops all 20 spectrogram array features (14 Omicron metadata features remaining)"
        }
    ]

    ablation_records = []
    per_class_ablation_records = {}
    baseline_macro_f1 = None

    for i, exp in enumerate(ablations):
        exp_name = exp["name"]
        subset_cols = exp["features"]
        n_feats = len(subset_cols)

        print(f"\n--- Running Ablation [{i+1}/{len(ablations)}]: {exp_name} ({n_feats} features) ---")
        t0 = time.time()

        # Extract subsets
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prepare_feature_matrices(
            train_df, val_df, test_df, feature_cols=subset_cols
        )

        # Train Random Forest using optimal hyperparameters (n_estimators=50, max_depth=None)
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=None,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        train_time = time.time() - t0

        # Evaluate on test set
        test_preds = rf.predict(X_te)
        metrics = compute_all_metrics(y_te, test_preds, class_names=class_names)

        acc = metrics["accuracy"] * 100
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]

        if baseline_macro_f1 is None:
            baseline_macro_f1 = macro_f1
            delta_f1 = 0.0
        else:
            delta_f1 = macro_f1 - baseline_macro_f1

        print(f"  Test Accuracy : {acc:.2f}% | Test Macro-F1: {macro_f1:.4f} | Delta Macro-F1: {delta_f1:+.4f} (Train: {train_time:.2f}s)")

        ablation_records.append({
            "Experiment": exp_name,
            "Num_Features": n_feats,
            "Accuracy (%)": round(acc, 2),
            "Macro-F1": round(macro_f1, 4),
            "Weighted-F1": round(weighted_f1, 4),
            "Delta_Macro_F1": round(delta_f1, 4),
            "Description": exp["description"]
        })

        per_class_ablation_records[exp_name] = metrics["per_class"]["F1-Score"].values

    # Convert to DataFrame
    ablation_df = pd.DataFrame(ablation_records)
    csv_path = os.path.join(tables_dir, "ablation_results.csv")
    ablation_df.to_csv(csv_path, index=False)
    print(f"\nAblation results table saved to: {csv_path}")

    # Save detailed per-class ablation breakdown
    per_class_abl_df = pd.DataFrame({"Class": class_names})
    for exp_name, f1_vals in per_class_ablation_records.items():
        per_class_abl_df[exp_name] = f1_vals
    per_class_csv = os.path.join(tables_dir, "per_class_ablation_breakdown.csv")
    per_class_abl_df.to_csv(per_class_csv, index=False)
    print(f"Per-class ablation breakdown saved to: {per_class_csv}")

    # Print summary
    print("\n" + "=" * 80)
    print(" SUMMARY OF FEATURE ABLATION EXPERIMENTS")
    print("=" * 80)
    print(ablation_df[['Experiment', 'Num_Features', 'Accuracy (%)', 'Macro-F1', 'Delta_Macro_F1']].to_string(index=False))

    # 5. Plot Ablation Impact Bar Chart
    plt.figure(figsize=(11, 6))
    colors = ['#1f77b4' if d == 0 else ('#d62728' if d < -0.05 else '#ff7f0e') for d in ablation_df['Delta_Macro_F1']]
    bars = plt.barh(ablation_df['Experiment'].iloc[::-1], ablation_df['Macro-F1'].iloc[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)

    # Reference baseline dashed line
    plt.axvline(x=baseline_macro_f1, color='#1f77b4', linestyle='--', linewidth=1.5, label=f"Full Model Macro-F1 ({baseline_macro_f1:.4f})")
    plt.xlim(0.70, 0.98)
    plt.xlabel("Test Set Macro-F1 Score", fontsize=11, fontweight='bold')
    plt.title("Impact of Feature-Group Ablations on LIGO Glitch Classification (Random Forest)", fontsize=12, fontweight='bold', pad=12)

    # Annotate delta values
    for bar, delta in zip(bars, ablation_df['Delta_Macro_F1'].iloc[::-1]):
        w = bar.get_width()
        txt = f" {w:.4f} (Ref)" if delta == 0 else f" {w:.4f} ({delta:+.4f})"
        plt.text(w + 0.003, bar.get_y() + bar.get_height() / 2, txt, va='center', fontsize=9, fontweight='bold')

    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "ablation_macro_f1_impact.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Ablation figure saved to: {fig_path}")

    return ablation_df, per_class_abl_df


def main():
    features_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")
    models_dir = os.path.join(project_root, "outputs", "models")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=" * 70)
    print(" STEP 10: FEATURE IMPORTANCE & ABLATION EXPERIMENTS")
    print("=" * 70)

    # 1. Load data and partitions
    df = pd.read_csv(features_csv)
    with open(splits_json, 'r', encoding='utf-8') as f:
        split_registry = json.load(f)

    default_split = split_registry['default_split']
    class_names = split_registry['classes']

    train_ids = set(default_split['train_ids'])
    val_ids = set(default_split['val_ids'])
    test_ids = set(default_split['test_ids'])

    train_df = df[df['gravityspy_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df['gravityspy_id'].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df['gravityspy_id'].isin(test_ids)].copy().reset_index(drop=True)

    print(f"Partitions loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

    # 2. Load trained optimal model
    rf_model_path = os.path.join(models_dir, "rf_best_model.joblib")
    if not os.path.exists(rf_model_path):
        print(f"Model not found at {rf_model_path}. Training Random Forest...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_feature_matrices(train_df, val_df, test_df)
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=None, max_features='sqrt', random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, rf_model_path)
    else:
        rf_model = joblib.load(rf_model_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_feature_matrices(train_df, val_df, test_df)

    # 3. Compute Feature Importances (MDI & Permutation)
    importance_df = compute_feature_importances(
        rf_model, X_test, y_test, ALL_FEATURE_COLS, class_names, tables_dir, figures_dir
    )

    # 4. Run Feature-Group Ablations
    ablation_df, per_class_abl_df = run_ablation_experiments(
        train_df, val_df, test_df, class_names, tables_dir, figures_dir
    )

    print("\n" + "=" * 70)
    print(" STEP 10 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
