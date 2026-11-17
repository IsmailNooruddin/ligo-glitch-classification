"""
Module: metadata_features.py
Purpose: Extracts and processes tabular Omicron trigger features and coordinates full batch feature caching.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
In addition to 2D spectrogram image features, LIGO's Omicron pipeline records physical parameters
(duration, peak frequency, SNR, bandwidth, amplitude). This module cleans and log-transforms these
physical properties, merges them with the 20 spectrogram features from `extractor.py`, and generates
a single consolidated table: `data/processed/engineered_features.csv`.

Once cached, all classical machine learning models (Decision Trees, Random Forests, ablations)
can be trained in mere seconds without re-reading the 3 GB HDF5 file.
"""

import os
import sys
import time
import h5py
import numpy as np
import pandas as pd

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.features.extractor import extract_spectrogram_features
from src.data.dataset import load_spectrogram


def extract_metadata_features(row):
    """
    Extracts raw and log-transformed physical trigger features from a single metadata row.

    Parameters:
    -----------
    row : pd.Series or dict
        A row from trainingset_v1d1_metadata.csv.

    Returns:
    --------
    feats : dict
        Clean dictionary of physical metadata measurements.
    """
    eps = 1e-8

    # Safe float conversions
    duration = float(row.get('duration', 0.0))
    peak_freq = float(row.get('peak_frequency', 0.0))
    central_freq = float(row.get('central_freq', 0.0))
    bandwidth = float(row.get('bandwidth', 0.0))
    amplitude = float(row.get('amplitude', 0.0))
    snr = float(row.get('snr', 0.0))

    feats = {
        # 1. Raw Physical Quantities
        'meta_duration': duration,
        'meta_peak_frequency': peak_freq,
        'meta_central_freq': central_freq,
        'meta_bandwidth': bandwidth,
        'meta_amplitude': amplitude,
        'meta_snr': snr,

        # 2. Log-Transformed Features (Compresses extreme exponential scales)
        # For example, amplitude ranges ~1e-22 and SNR ranges from 7.5 to 1000+
        'meta_log_duration': float(np.log10(max(duration, 1e-4))),
        'meta_log_peak_frequency': float(np.log10(max(peak_freq, 1.0))),
        'meta_log_central_freq': float(np.log10(max(central_freq, 1.0))),
        'meta_log_bandwidth': float(np.log10(max(bandwidth, 0.1))),
        'meta_log_snr': float(np.log10(max(snr, 1.0))),
        'meta_log_amplitude': float(np.log10(max(amplitude, 1e-25))),

        # 3. Derived Physical Ratios
        # Q-factor approximation: center frequency divided by bandwidth
        'meta_q_factor_approx': float(central_freq / (bandwidth + eps)),
        # Relative bandwidth: bandwidth divided by peak frequency
        'meta_bandwidth_ratio': float(bandwidth / (peak_freq + eps))
    }

    return feats


def build_full_engineered_features(metadata_csv_path, h5_path, output_csv_path, view="1.0s"):
    """
    Extracts spectrogram features and metadata features for all glitches in the dataset,
    combines them with ground-truth identifiers, and saves the consolidated table to CSV.

    Parameters:
    -----------
    metadata_csv_path : str
        Path to 'data/processed/shared_classes_metadata.csv'.
    h5_path : str
        Path to 'trainingsetv1d1.h5'.
    output_csv_path : str
        Destination path for 'data/processed/engineered_features.csv'.
    view : str, default="1.0s"
        Which spectrogram duration view to extract (1.0s is the standard baseline view).
    """
    start_time = time.time()

    print(f"Loading metadata from: {metadata_csv_path}")
    df = pd.read_csv(metadata_csv_path)
    total_samples = len(df)
    print(f"Total glitches to process: {total_samples}")
    print(f"Reading spectrograms from: {h5_path} (View: {view})")

    records = []

    # Open HDF5 file once and iterate through samples
    with h5py.File(h5_path, "r") as h5_f:
        for idx in range(total_samples):
            row = df.iloc[idx]
            glitch_id = row['gravityspy_id']
            label = row['label']
            ifo = row['ifo']
            sample_type = row['sample_type']

            # 1. Base metadata identifiers
            item_data = {
                'gravityspy_id': glitch_id,
                'label': label,
                'ifo': ifo,
                'sample_type': sample_type
            }

            # 2. Extract 2D spectrogram features
            try:
                spec_arr = load_spectrogram(h5_f, label, sample_type, glitch_id, view=view)
                spec_feats = extract_spectrogram_features(spec_arr, prefix="spec_")
                item_data.update(spec_feats)
            except Exception as e:
                print(f"Warning: Failed loading spectrogram for {glitch_id} ({e})")
                continue

            # 3. Extract physical Omicron metadata features
            meta_feats = extract_metadata_features(row)
            item_data.update(meta_feats)

            records.append(item_data)

            # Print progress every 1,000 samples
            if (idx + 1) % 1000 == 0 or (idx + 1) == total_samples:
                elapsed = time.time() - start_time
                print(f"Processed {idx + 1}/{total_samples} samples ({elapsed:.1f}s elapsed)...")

    # Convert all records into a single DataFrame
    features_df = pd.DataFrame(records)

    # Verify data integrity
    print("\n--- Feature Extraction Integrity Check ---")
    print(f"Total Rows Extracted : {len(features_df)}")
    print(f"Total Columns        : {features_df.shape[1]}")
    null_count = features_df.isnull().sum().sum()
    print(f"Total Missing/Nulls  : {null_count}")

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Save to CSV
    features_df.to_csv(output_csv_path, index=False)
    total_elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
    print(f"Successfully saved engineered features to: {output_csv_path}")
    print(f"File size: {file_size_mb:.2f} MB | Total runtime: {total_elapsed:.1f}s")

    return features_df


if __name__ == "__main__":
    # When this script is run directly from the command line:
    input_meta = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")
    input_h5 = os.path.join(project_root, "trainingsetv1d1.h5")
    output_csv = os.path.join(project_root, "data", "processed", "engineered_features.csv")

    build_full_engineered_features(input_meta, input_h5, output_csv, view="1.0s")
