

import os
import sys
import json
import time
import h5py
import numpy as np
import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.baselines import MajorityClassBaseline, RawPixelDecisionTree
from src.data.dataset import load_spectrogram
from src.evaluation.metrics import compute_all_metrics, print_metrics_summary


def load_dataset_and_splits(features_csv_path, splits_json_path):
    """
    Loads engineered features and partitions into train, validation, and test sets.
    """
    df = pd.read_csv(features_csv_path)
    with open(splits_json_path, 'r', encoding='utf-8') as f:
        split_registry = json.load(f)

    # Use the default Seed 42 split
    default_split = split_registry['default_split']
    train_ids = set(default_split['train_ids'])
    val_ids = set(default_split['val_ids'])
    test_ids = set(default_split['test_ids'])

    # Separate into DataFrames
    train_df = df[df['gravityspy_id'].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df['gravityspy_id'].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df['gravityspy_id'].isin(test_ids)].copy().reset_index(drop=True)

    print(f"Dataset partitions loaded successfully:")
    print(f"  Training samples   : {len(train_df)}")
    print(f"  Validation samples : {len(val_df)}")
    print(f"  Testing samples    : {len(test_df)}")

    return train_df, val_df, test_df, split_registry['classes']


def load_flattened_pixel_arrays(h5_path, metadata_df, view="1.0s"):
    """
    Loads and flattens 2D spectrogram matrices (140 x 170) into 23,800-dimensional 1D vectors.

    Parameters:
    -----------
    h5_path : str
        Path to trainingsetv1d1.h5.
    metadata_df : pd.DataFrame
        DataFrame containing 'label', 'sample_type', 'gravityspy_id'.
    view : str, default="1.0s"
        Spectrogram duration view to load.

    Returns:
    --------
    X_pixels : np.ndarray
        Array of shape (N, 23800), dtype float32.
    y : np.ndarray
        Array of label strings of length N.
    """
    n_samples = len(metadata_df)
    pixel_dim = 140 * 170  # 23,800 features per glitch
    X_pixels = np.zeros((n_samples, pixel_dim), dtype=np.float32)
    y = metadata_df['label'].values

    print(f"Loading {n_samples} raw spectrogram arrays from HDF5 (view: {view})...")
    start_t = time.time()

    with h5py.File(h5_path, "r") as h5_f:
        for i in range(n_samples):
            row = metadata_df.iloc[i]
            # Load (140, 170) array and flatten to 1D
            arr = load_spectrogram(h5_f, row['label'], row['sample_type'], row['gravityspy_id'], view=view)
            X_pixels[i, :] = arr.ravel()

            if (i + 1) % 1500 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - start_t
                print(f"  Loaded {i + 1}/{n_samples} arrays ({elapsed:.1f}s elapsed)...")

    return X_pixels, y


def run_majority_baseline(train_df, test_df, class_names, tables_dir):
    """
    Trains and evaluates the Majority Class baseline.
    """
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    model = MajorityClassBaseline()
    model.fit(train_df, y_train)

    majority_label = model.majority_class_
    y_pred = model.predict(test_df)

    metrics = compute_all_metrics(y_test, y_pred, class_names=class_names)
    print_metrics_summary("Baseline 1: Majority Class Classifier", metrics)

    # Save to Benchmark Table
    os.makedirs(tables_dir, exist_ok=True)
    benchmark_file = os.path.join(tables_dir, "benchmark_comparison.csv")

    benchmark_entry = {
        "Model": "Baseline 1: Majority Class",
        "Accuracy (%)": round(metrics["accuracy"] * 100, 2),
        "Macro-F1": round(metrics["macro_f1"], 4),
        "Weighted-F1": round(metrics["weighted_f1"], 4),
        "Description": f"Always predicts '{majority_label}' (most common training class)"
    }

    if os.path.exists(benchmark_file):
        bench_df = pd.read_csv(benchmark_file)
        bench_df = bench_df[bench_df["Model"] != benchmark_entry["Model"]]
        bench_df = pd.concat([bench_df, pd.DataFrame([benchmark_entry])], ignore_index=True)
    else:
        bench_df = pd.DataFrame([benchmark_entry])

    bench_df.to_csv(benchmark_file, index=False)

    # Save to Per-Class F1 Table
    per_class_file = os.path.join(tables_dir, "per_class_f1_summary.csv")
    per_class_df = metrics["per_class"].copy()
    per_class_df = per_class_df.rename(columns={"F1-Score": "Majority_Baseline_F1"})
    per_class_df = per_class_df[["Class", "Support", "Majority_Baseline_F1"]]
    per_class_df.to_csv(per_class_file, index=False)

    return metrics


def run_raw_pixel_tree_baseline(train_df, test_df, h5_path, class_names, tables_dir):
    """
    Trains and evaluates Baseline 2: Raw-Pixel Decision Tree on 23,800 flattened pixel numbers.
    """
    print("\n" + "="*66)
    print(" TRAINING BASELINE 2: RAW-PIXEL DECISION TREE (23,800 DIMENSIONS)")
    print("="*66)

    # 1. Load flattened pixel vectors for Train and Test sets
    X_train_pixels, y_train = load_flattened_pixel_arrays(h5_path, train_df, view="1.0s")
    X_test_pixels, y_test = load_flattened_pixel_arrays(h5_path, test_df, view="1.0s")

    print(f"Feature matrix dimensions:")
    print(f"  X_train_pixels: {X_train_pixels.shape} ({X_train_pixels.nbytes / (1024**2):.1f} MB in RAM)")
    print(f"  X_test_pixels : {X_test_pixels.shape} ({X_test_pixels.nbytes / (1024**2):.1f} MB in RAM)")

    # 2. Train Decision Tree directly on raw pixel numbers
    print("\nFitting DecisionTree(max_depth=10, random_state=42) on raw pixels...")
    tree_start = time.time()
    pixel_tree = RawPixelDecisionTree(max_depth=10, random_state=42)
    pixel_tree.fit(X_train_pixels, y_train)
    fit_time = time.time() - tree_start
    print(f"Training completed in: {fit_time:.2f} seconds")

    # 3. Evaluate training performance (check for memorization/overfitting)
    train_preds = pixel_tree.predict(X_train_pixels)
    train_acc = np.mean(train_preds == y_train)
    print(f"Training Set Accuracy : {train_acc * 100:.2f}% (High memorization)")

    # 4. Evaluate test performance
    test_preds = pixel_tree.predict(X_test_pixels)
    test_metrics = compute_all_metrics(y_test, test_preds, class_names=class_names)
    print_metrics_summary("Baseline 2: Raw-Pixel Decision Tree", test_metrics)

    # 5. Append to Benchmark Comparison Table
    benchmark_file = os.path.join(tables_dir, "benchmark_comparison.csv")
    bench_entry = {
        "Model": "Baseline 2: Raw-Pixel Decision Tree",
        "Accuracy (%)": round(test_metrics["accuracy"] * 100, 2),
        "Macro-F1": round(test_metrics["macro_f1"], 4),
        "Weighted-F1": round(test_metrics["weighted_f1"], 4),
        "Description": "Decision Tree (max_depth=10) on 23,800 raw pixels (overfitting on pixel noise)"
    }

    bench_df = pd.read_csv(benchmark_file)
    bench_df = bench_df[bench_df["Model"] != bench_entry["Model"]]
    bench_df = pd.concat([bench_df, pd.DataFrame([bench_entry])], ignore_index=True)
    bench_df.to_csv(benchmark_file, index=False)
    print(f"Benchmark table updated at: {benchmark_file}")

    # 6. Append to Per-Class F1 Table
    per_class_file = os.path.join(tables_dir, "per_class_f1_summary.csv")
    if os.path.exists(per_class_file):
        per_class_df = pd.read_csv(per_class_file)
        per_class_df["Raw_Pixel_Tree_F1"] = test_metrics["per_class"]["F1-Score"].values
        per_class_df.to_csv(per_class_file, index=False)
        print(f"Per-class F1 table updated with Raw_Pixel_Tree_F1 at: {per_class_file}")

    return test_metrics


if __name__ == "__main__":
    features_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")
    splits_json = os.path.join(project_root, "data", "processed", "split_indices.json")
    h5_file = os.path.join(project_root, "trainingsetv1d1.h5")
    tables_dir = os.path.join(project_root, "outputs", "tables")

    train_df, val_df, test_df, classes = load_dataset_and_splits(features_csv, splits_json)

    # 1. Evaluate Baseline 1: Majority Class
    run_majority_baseline(train_df, test_df, classes, tables_dir)

    # 2. Evaluate Baseline 2: Raw-Pixel Decision Tree
    run_raw_pixel_tree_baseline(train_df, test_df, h5_file, classes, tables_dir)
