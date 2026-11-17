"""
Module: split_utils.py
Purpose: Creates deterministic, leak-free train/validation/test data splits.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Each glitch event is captured across 4 duration zoom views (0.5s, 1.0s, 2.0s, 4.0s).
To prevent data leakage, ALL views of a single glitch (identified by `gravityspy_id`)
must be assigned to the exact same partition (train, validation, or test).
We also generate cross-detector partitions (Hanford H1 vs Livingston L1) and multi-seed splits
to measure statistical variability across runs (mean ± standard deviation).
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def create_stratified_glitch_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits a metadata DataFrame into Train, Validation, and Test sets based on
    unique `gravityspy_id`, stratified by the glitch `label`.

    Parameters:
    -----------
    df : pd.DataFrame
        The metadata table containing 'gravityspy_id' and 'label'.
    test_size : float, default=0.15
        Fraction of data reserved for the final test set (15%).
    val_size : float, default=0.15
        Fraction of data reserved for validation / tuning (15%).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    split_dict : dict
        Dictionary with keys 'train_ids', 'val_ids', and 'test_ids'.
    """
    # Step 1: Ensure each row is a unique glitch ID
    unique_glitches = df[['gravityspy_id', 'label']].drop_duplicates(subset=['gravityspy_id']).copy()

    # Step 2: Separate out the Test set first (e.g. 15% of total)
    train_val_df, test_df = train_test_split(
        unique_glitches,
        test_size=test_size,
        stratify=unique_glitches['label'],
        random_state=random_state
    )

    # Step 3: From the remaining (85%), separate out the Validation set
    # To get exactly val_size (15%) of the original total:
    # adjusted_val_fraction = val_size / (1.0 - test_size)
    adjusted_val_ratio = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df['label'],
        random_state=random_state
    )

    # Step 4: Extract the lists of IDs for each split
    split_dict = {
        "train_ids": train_df['gravityspy_id'].tolist(),
        "val_ids": val_df['gravityspy_id'].tolist(),
        "test_ids": test_df['gravityspy_id'].tolist()
    }

    return split_dict


def create_detector_splits(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Creates individual stratified splits specifically for Hanford (H1) and Livingston (L1).
    This supports Option B: Cross-Detector Testing.

    Returns:
    --------
    detector_splits : dict
        Contains 'H1' and 'L1' sub-dictionaries with train, val, and test IDs.
    """
    h1_df = df[df['ifo'] == 'H1'].copy()
    l1_df = df[df['ifo'] == 'L1'].copy()

    h1_split = create_stratified_glitch_split(h1_df, test_size=test_size, val_size=val_size, random_state=random_state)
    l1_split = create_stratified_glitch_split(l1_df, test_size=test_size, val_size=val_size, random_state=random_state)

    return {
        "H1": h1_split,
        "L1": l1_split
    }


def verify_split_leakage(split_dict):
    """
    Mathematically verifies that no gravityspy_id appears in more than one partition.

    Returns:
    --------
    is_valid : bool
        True if all partitions are completely disjoint, False otherwise.
    """
    train_set = set(split_dict["train_ids"])
    val_set = set(split_dict["val_ids"])
    test_set = set(split_dict["test_ids"])

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    has_leakage = (len(train_val_overlap) > 0) or (len(train_test_overlap) > 0) or (len(val_test_overlap) > 0)

    if has_leakage:
        print("ERROR: Data leakage detected between partitions!")
        print(f"Train & Val Overlap : {len(train_val_overlap)}")
        print(f"Train & Test Overlap: {len(train_test_overlap)}")
        print(f"Val & Test Overlap  : {len(val_test_overlap)}")
        return False

    print("VERIFICATION PASSED: All splits are 100% disjoint. Zero view-leakage confirmed.")
    return True


def generate_and_save_all_splits(metadata_csv_path, output_json_path, seeds=[42, 43, 44, 45, 46]):
    """
    Loads shared classes metadata, generates default, detector-specific, and multi-seed splits,
    verifies them, and saves everything to split_indices.json.
    """
    print(f"Loading shared metadata from: {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)
    total_samples = len(df)
    print(f"Total samples to partition: {total_samples}")

    # 1. Create Default Master Split (Seed 42)
    default_split = create_stratified_glitch_split(df, test_size=0.15, val_size=0.15, random_state=42)
    print("\n--- Default Pooled Split (Seed 42) ---")
    print(f"Train Glitches : {len(default_split['train_ids'])} ({len(default_split['train_ids'])/total_samples*100:.1f}%)")
    print(f"Val Glitches   : {len(default_split['val_ids'])} ({len(default_split['val_ids'])/total_samples*100:.1f}%)")
    print(f"Test Glitches  : {len(default_split['test_ids'])} ({len(default_split['test_ids'])/total_samples*100:.1f}%)")
    verify_split_leakage(default_split)

    # 2. Create Detector-Specific Splits (Hanford H1 & Livingston L1)
    detector_splits = create_detector_splits(df, test_size=0.15, val_size=0.15, random_state=42)
    print("\n--- Detector-Specific Splits (Seed 42) ---")
    print(f"Hanford (H1)    -> Train: {len(detector_splits['H1']['train_ids'])}, Val: {len(detector_splits['H1']['val_ids'])}, Test: {len(detector_splits['H1']['test_ids'])}")
    print(f"Livingston (L1) -> Train: {len(detector_splits['L1']['train_ids'])}, Val: {len(detector_splits['L1']['val_ids'])}, Test: {len(detector_splits['L1']['test_ids'])}")
    verify_split_leakage(detector_splits["H1"])
    verify_split_leakage(detector_splits["L1"])

    # 3. Create Multi-Seed Splits (to measure mean ± std dev across runs)
    multi_seed_splits = {}
    for s in seeds:
        seed_split = create_stratified_glitch_split(df, test_size=0.15, val_size=0.15, random_state=s)
        multi_seed_splits[str(s)] = seed_split
    print(f"\nGenerated multi-seed splits for seeds: {seeds}")

    # 4. Package all split configurations together
    full_split_registry = {
        "dataset": "shared_classes_metadata.csv",
        "total_samples": total_samples,
        "num_classes": int(df['label'].nunique()),
        "classes": sorted(df['label'].unique().tolist()),
        "proportions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "default_split": default_split,
        "detector_splits": detector_splits,
        "multi_seed_splits": multi_seed_splits
    }

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(full_split_registry, f, indent=2)

    print(f"\nAll split configurations saved successfully to: {output_json_path}")


if __name__ == "__main__":
    # When this script is run directly from the command line:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    metadata_path = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    output_json = os.path.join(project_root, "data", "processed", "split_indices.json")

    generate_and_save_all_splits(metadata_path, output_json, seeds=[42, 43, 44, 45, 46])
