"""
Module: class_selector.py
Purpose: Identifies and filters shared glitch classes between Hanford (H1) and Livingston (L1).
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
In the Gravity Spy dataset, certain noise glitches only occur at one detector (e.g. Whistle at Livingston,
Light Modulation at Hanford). For valid Cross-Detector Testing (Option B), the model must only be trained
and evaluated on classes that exist with sufficient examples at BOTH detector sites.
"""

import os
import pandas as pd


def get_shared_classes(metadata_df, min_samples_per_detector=25):
    """
    Finds all glitch classes that have at least `min_samples_per_detector` samples
    in Hanford (H1) AND in Livingston (L1).

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        The full dataset metadata table containing 'label' and 'ifo' columns.
    min_samples_per_detector : int, default=25
        The minimum number of glitch events a class must have at each detector.

    Returns:
    --------
    shared_classes : list of str
        Alphabetically sorted list of shared class names that meet the threshold.
    summary_table : pd.DataFrame
        A table showing counts for H1, L1, Total, and whether each class qualified.
    """
    # Step 1: Count how many times each class appears at each detector (H1 vs L1)
    # pd.crosstab creates a table where rows are classes and columns are 'H1' and 'L1'
    crosstab = pd.crosstab(metadata_df['label'], metadata_df['ifo'])

    # Step 2: Make sure both 'H1' and 'L1' columns exist in case a subset is passed
    if 'H1' not in crosstab.columns:
        crosstab['H1'] = 0
    if 'L1' not in crosstab.columns:
        crosstab['L1'] = 0

    # Step 3: Compute total count for each class
    crosstab['Total'] = crosstab['H1'] + crosstab['L1']

    # Step 4: Check which classes satisfy the minimum threshold at BOTH detectors
    is_shared = (crosstab['H1'] >= min_samples_per_detector) & (crosstab['L1'] >= min_samples_per_detector)
    crosstab['Is_Shared'] = is_shared

    # Step 5: Extract the list of qualifying class names
    shared_classes = crosstab[is_shared].index.tolist()
    shared_classes.sort()

    # Sort the summary table so the biggest classes appear at the top
    summary_table = crosstab.sort_values(by='Total', ascending=False)

    return shared_classes, summary_table


def filter_dataset_by_shared_classes(metadata_df, shared_classes):
    """
    Filters the metadata table to keep only the rows belonging to the shared classes.

    Parameters:
    -----------
    metadata_df : pd.DataFrame
        The full metadata DataFrame.
    shared_classes : list of str
        List of allowed class labels.

    Returns:
    --------
    filtered_df : pd.DataFrame
        A new DataFrame containing only rows whose 'label' is in `shared_classes`.
    """
    # Use pandas .isin() to create a True/False filter for shared classes
    keep_mask = metadata_df['label'].isin(shared_classes)

    # Filter and make a fresh copy to prevent SettingWithCopy warnings
    filtered_df = metadata_df[keep_mask].copy().reset_index(drop=True)

    return filtered_df


def save_shared_classes_metadata(input_csv_path, output_csv_path, min_samples=25):
    """
    Loads raw metadata, filters for shared classes, and saves the cleaned CSV.

    Parameters:
    -----------
    input_csv_path : str
        Path to 'trainingset_v1d1_metadata.csv'.
    output_csv_path : str
        Destination path where 'shared_classes_metadata.csv' will be written.
    min_samples : int, default=25
        Minimum sample threshold per detector.

    Returns:
    --------
    filtered_df : pd.DataFrame
        The processed DataFrame that was saved.
    """
    print(f"Loading raw metadata from: {input_csv_path}")
    raw_df = pd.read_csv(input_csv_path)
    total_raw_rows = len(raw_df)

    # Find the shared classes
    shared_classes, summary_table = get_shared_classes(raw_df, min_samples_per_detector=min_samples)

    print(f"\n--- Shared Classes Analysis (Threshold: >= {min_samples} samples per detector) ---")
    print(f"Total Unique Classes in Raw Data : {len(summary_table)}")
    print(f"Classes Meeting Shared Threshold : {len(shared_classes)}")

    # Filter the dataset
    filtered_df = filter_dataset_by_shared_classes(raw_df, shared_classes)
    total_filtered_rows = len(filtered_df)
    retention_pct = (total_filtered_rows / total_raw_rows) * 100

    print(f"\n--- Data Retention Summary ---")
    print(f"Original Glitches : {total_raw_rows}")
    print(f"Retained Glitches : {total_filtered_rows} ({retention_pct:.2f}% of full dataset)")
    print(f"Excluded Glitches : {total_raw_rows - total_filtered_rows} ({100 - retention_pct:.2f}%)")

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save to CSV
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"\nFiltered dataset successfully saved to: {output_csv_path}")

    return filtered_df


if __name__ == "__main__":
    # When this script is run directly from the command line:
    # 1. Determine base path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # 2. Define input and output file paths
    input_file = os.path.join(project_root, "trainingset_v1d1_metadata.csv")
    output_file = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")

    # 3. Execute the filtering and saving pipeline
    save_shared_classes_metadata(input_file, output_file, min_samples=25)
