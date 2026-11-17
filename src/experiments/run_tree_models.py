"""
Module: run_tree_models.py
Purpose: Complete training, validation tuning, and test evaluation for Feature Decision Tree and Random Forest.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Key Milestones Accomplished:
1. Load 34 engineered features for default Seed 42 partition (4,437 train, 952 val, 951 test).
2. Tune Feature-Based Decision Tree over max_depth and class_weight using validation Macro-F1.
3. Tune Feature-Based Random Forest over n_estimators, max_depth, and class_weight using validation Macro-F1.
4. Evaluate both optimal models on the unseen test set.
5. Save the trained Random Forest model artifact to outputs/models/rf_best_model.joblib.
6. Generate publication-quality Confusion Matrix heatmap at outputs/figures/confusion_matrix_rf.png.
7. Update benchmark comparison and per-class F1 tables in outputs/tables/.
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

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.tree_models import (
    ALL_FEATURE_COLS,
    prepare_feature_matrices,
    tune_decision_tree,
    tune_random_forest,
    train_decision_tree,
    train_random_forest
)
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


def plot_and_save_confusion_matrix(cm, class_names, output_path, model_name="Random Forest"):
    """
    Plots a normalized confusion matrix heatmap and saves it as a high-res PNG.
    """
    # Normalize by true row sum to show recall / class-accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Recall'}
    )
    plt.title(f"Confusion Matrix (Normalized) - {model_name}\n(Evaluated on Unseen Test Set, N=951)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted Glitch Class", fontsize=11, fontweight='bold')
    plt.ylabel("True Glitch Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def run_tree_experiments():
    features_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    tables_dir = os.path.join(project_root, "outputs", "tables")
    figures_dir = os.path.join(project_root, "outputs", "figures")
    models_dir = os.path.join(project_root, "outputs", "models")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print(" STEP 9: FEATURE-BASED DECISION TREE & RANDOM FOREST EXPERIMENTS")
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

    print(f"Partition summary:")
    print(f"  Training samples   : {len(train_df)}")
    print(f"  Validation samples : {len(val_df)}")
    print(f"  Testing samples    : {len(test_df)}")
    print(f"  Number of features : {len(ALL_FEATURE_COLS)} (20 spectrogram + 14 Omicron metadata)")

    # 2. Prepare feature matrices
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_feature_matrices(
        train_df, val_df, test_df, feature_cols=ALL_FEATURE_COLS
    )

    # -------------------------------------------------------------
    # 3. Model 1: Feature-Based Decision Tree
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [1/2] TUNING & EVALUATING FEATURE-BASED DECISION TREE")
    print("-" * 70)

    best_dt, dt_params, dt_tuning_df = tune_decision_tree(
        X_train, y_train, X_val, y_val, class_names,
        depth_options=[4, 6, 8, 10, 12, 15, None],
        class_weight_options=[None, 'balanced'],
        min_samples_split_options=[2, 5, 10],
        random_state=42
    )

    # Evaluate best Decision Tree on test set
    t0 = time.time()
    y_test_pred_dt = best_dt.predict(X_test)
    dt_eval_time = time.time() - t0

    dt_test_metrics = compute_all_metrics(y_test, y_test_pred_dt, class_names=class_names)
    print_metrics_summary(f"Feature Decision Tree (depth={dt_params['max_depth']}, cw={dt_params['class_weight']})", dt_test_metrics)

    # -------------------------------------------------------------
    # 4. Model 2: Feature-Based Random Forest
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print(" [2/2] TUNING & EVALUATING FEATURE-BASED RANDOM FOREST")
    print("-" * 70)

    best_rf, rf_params, rf_tuning_df = tune_random_forest(
        X_train, y_train, X_val, y_val, class_names,
        n_estimators_options=[50, 100, 200],
        depth_options=[8, 12, 16, 20, None],
        class_weight_options=[None, 'balanced', 'balanced_subsample'],
        random_state=42,
        n_jobs=-1
    )

    # Evaluate best Random Forest on test set
    t0 = time.time()
    y_test_pred_rf = best_rf.predict(X_test)
    rf_eval_time = time.time() - t0

    rf_test_metrics = compute_all_metrics(y_test, y_test_pred_rf, class_names=class_names)
    print_metrics_summary(f"Feature Random Forest (n_est={rf_params['n_estimators']}, depth={rf_params['max_depth']}, cw={rf_params['class_weight']})", rf_test_metrics)

    # -------------------------------------------------------------
    # 5. Save Artifacts: Best Model & Confusion Matrix
    # -------------------------------------------------------------
    rf_model_path = os.path.join(models_dir, "rf_best_model.joblib")
    joblib.dump(best_rf, rf_model_path)
    print(f"\nSaved optimal Random Forest model to: {rf_model_path}")

    # Remove placeholder file if it exists
    placeholder_rf = os.path.join(models_dir, "PLACEHOLDER_rf_best_model_LATER.joblib.txt")
    if os.path.exists(placeholder_rf):
        os.remove(placeholder_rf)
        print(f"Removed placeholder file: {placeholder_rf}")

    # Plot and save confusion matrix
    cm_path = os.path.join(figures_dir, "confusion_matrix_rf.png")
    plot_and_save_confusion_matrix(rf_test_metrics['confusion_matrix'], class_names, cm_path, model_name="Random Forest (34 Features)")

    # -------------------------------------------------------------
    # 6. Update Benchmark Comparison Table
    # -------------------------------------------------------------
    benchmark_file = os.path.join(tables_dir, "benchmark_comparison.csv")
    bench_df = pd.read_csv(benchmark_file) if os.path.exists(benchmark_file) else pd.DataFrame()

    new_entries = [
        {
            "Model": "Feature-Based Decision Tree",
            "Accuracy (%)": round(dt_test_metrics["accuracy"] * 100, 2),
            "Macro-F1": round(dt_test_metrics["macro_f1"], 4),
            "Weighted-F1": round(dt_test_metrics["weighted_f1"], 4),
            "Description": f"Tuned Decision Tree (max_depth={dt_params['max_depth']}, class_weight={dt_params['class_weight']}) on 34 engineered features"
        },
        {
            "Model": "Feature-Based Random Forest",
            "Accuracy (%)": round(rf_test_metrics["accuracy"] * 100, 2),
            "Macro-F1": round(rf_test_metrics["macro_f1"], 4),
            "Weighted-F1": round(rf_test_metrics["weighted_f1"], 4),
            "Description": f"Tuned Random Forest (n_est={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}, class_weight={rf_params['class_weight']}) on 34 engineered features"
        }
    ]

    for entry in new_entries:
        if not bench_df.empty and entry["Model"] in bench_df["Model"].values:
            bench_df = bench_df[bench_df["Model"] != entry["Model"]]
        bench_df = pd.concat([bench_df, pd.DataFrame([entry])], ignore_index=True)

    bench_df.to_csv(benchmark_file, index=False)
    print(f"\nUpdated benchmark table at: {benchmark_file}")
    print("\nCURRENT BENCHMARK COMPARISON:")
    print(bench_df.to_string(index=False))

    # -------------------------------------------------------------
    # 7. Update Per-Class F1 Summary Table
    # -------------------------------------------------------------
    per_class_file = os.path.join(tables_dir, "per_class_f1_summary.csv")
    if os.path.exists(per_class_file):
        per_class_df = pd.read_csv(per_class_file)
    else:
        per_class_df = pd.DataFrame({"Class": class_names})

    per_class_df["Feature_Tree_F1"] = dt_test_metrics["per_class"]["F1-Score"].values
    per_class_df["Random_Forest_F1"] = rf_test_metrics["per_class"]["F1-Score"].values
    per_class_df.to_csv(per_class_file, index=False)
    print(f"\nUpdated per-class F1 table at: {per_class_file}")
    print("\nPER-CLASS F1 SUMMARY:")
    print(per_class_df.to_string(index=False))

    # Also save tuning logs for transparency
    dt_tuning_path = os.path.join(tables_dir, "dt_hyperparameter_tuning.csv")
    dt_tuning_df.to_csv(dt_tuning_path, index=False)

    rf_tuning_path = os.path.join(tables_dir, "rf_hyperparameter_tuning.csv")
    rf_tuning_df.to_csv(rf_tuning_path, index=False)
    print(f"\nSaved hyperparameter tuning logs to outputs/tables/")

    print("\nStep 9 experiment execution completed successfully!")


if __name__ == "__main__":
    run_tree_experiments()
