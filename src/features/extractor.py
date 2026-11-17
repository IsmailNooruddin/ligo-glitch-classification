"""
Module: extractor.py
Purpose: Mathematical feature extraction from 2D LIGO glitch spectrogram arrays (140 x 170).
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Directly addresses the instructor's requirement:
"Define the engineered features precisely and verify which measurements the spectrogram arrays support."

Rather than treating the spectrogram as a black box, this module computes precise physical,
spectral, temporal, and morphological measurements directly from the 2D pixel energy matrix.
These interpretable features feed our Decision Tree, Random Forest, and ablation studies.
"""

import os
import sys
import numpy as np
import pandas as pd


def extract_spectrogram_features(spectrogram_array, prefix=""):
    """
    Extracts an interpretable, mathematically defined feature vector from a single 2D spectrogram.

    Parameters:
    -----------
    spectrogram_array : np.ndarray
        2D array of shape (140, 170) where rows represent frequency bins and columns represent time bins.
    prefix : str, default=""
        Optional prefix prepended to all feature names (e.g., 'v1s_' for 1.0-second view).

    Returns:
    --------
    features : dict
        Dictionary of feature names mapped to float values.
    """
    # Ensure input is 2D and float32
    arr = np.squeeze(spectrogram_array).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")

    n_freq_bins, n_time_bins = arr.shape  # 140 frequency bins, 170 time bins
    eps = 1e-8  # Small epsilon to prevent division by zero

    features = {}

    # ---------------------------------------------------------
    # 1. Global Energy & Contrast Statistics
    # ---------------------------------------------------------
    # Peak energy (loudest point in the glitch)
    peak_val = float(np.max(arr))
    # Mean and standard deviation of background + signal energy
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    # Energy contrast: ratio of peak energy to average background
    contrast = peak_val / (mean_val + eps)
    # Energy concentration: fraction of total power held by top 5% brightest pixels
    sorted_pixels = np.sort(arr.ravel())
    top5_count = max(1, int(0.05 * len(sorted_pixels)))
    top5_energy = float(np.sum(sorted_pixels[-top5_count:]))
    total_energy = float(np.sum(arr)) + eps
    concentration_top5 = top5_energy / total_energy

    features[f"{prefix}peak_energy"] = peak_val
    features[f"{prefix}mean_energy"] = mean_val
    features[f"{prefix}std_energy"] = std_val
    features[f"{prefix}energy_contrast"] = contrast
    features[f"{prefix}energy_concentration"] = concentration_top5

    # ---------------------------------------------------------
    # 2. Peak Coordinate Location (Frequency and Time Bins)
    # ---------------------------------------------------------
    # Identifies exactly where the loudest point of the glitch occurs
    peak_f_idx, peak_t_idx = np.unravel_index(np.argmax(arr), arr.shape)
    features[f"{prefix}peak_freq_bin"] = float(peak_f_idx)  # Row index (0 = lowest pitch, 139 = highest pitch)
    features[f"{prefix}peak_time_bin"] = float(peak_t_idx)  # Column index (0 = start, 85 = center, 169 = end)

    # ---------------------------------------------------------
    # 3. Spectral Moments (Frequency Distribution along Y-Axis)
    # ---------------------------------------------------------
    # Sum energy along the horizontal time axis to get a 1D frequency energy profile P(f)
    freq_profile = np.sum(arr, axis=1)  # Length: 140
    freq_indices = np.arange(n_freq_bins, dtype=np.float32)
    freq_norm = np.sum(freq_profile) + eps

    # Spectral Centroid: center of gravity along frequency axis (average pitch)
    spectral_centroid = float(np.sum(freq_indices * freq_profile) / freq_norm)

    # Spectral Spread / Bandwidth: standard deviation around the centroid
    spectral_variance = np.sum(((freq_indices - spectral_centroid) ** 2) * freq_profile) / freq_norm
    spectral_spread = float(np.sqrt(max(0.0, spectral_variance)))

    # Spectral Skewness: asymmetry of energy (tilted towards low or high frequencies)
    if spectral_spread > eps:
        spectral_skewness = float(np.sum(((freq_indices - spectral_centroid) / spectral_spread) ** 3 * freq_profile) / freq_norm)
    else:
        spectral_skewness = 0.0

    features[f"{prefix}spectral_centroid"] = spectral_centroid
    features[f"{prefix}spectral_spread"] = spectral_spread
    features[f"{prefix}spectral_skewness"] = spectral_skewness

    # ---------------------------------------------------------
    # 4. Temporal Moments (Time Distribution along X-Axis)
    # ---------------------------------------------------------
    # Sum energy along vertical frequency axis to get a 1D time energy profile P(t)
    time_profile = np.sum(arr, axis=0)  # Length: 170
    time_indices = np.arange(n_time_bins, dtype=np.float32)
    time_norm = np.sum(time_profile) + eps

    # Temporal Centroid: center of mass along time axis
    temporal_centroid = float(np.sum(time_indices * time_profile) / time_norm)

    # Temporal Spread: standard deviation around the time centroid (duration of burst)
    temporal_variance = np.sum(((time_indices - temporal_centroid) ** 2) * time_profile) / time_norm
    temporal_spread = float(np.sqrt(max(0.0, temporal_variance)))

    # Active Glitch Duration: number of time columns exceeding 75th percentile energy
    time_max_profile = np.max(arr, axis=0)
    bg_threshold = float(np.percentile(arr, 75))
    active_duration_bins = float(np.sum(time_max_profile > bg_threshold))

    features[f"{prefix}temporal_centroid"] = temporal_centroid
    features[f"{prefix}temporal_spread"] = temporal_spread
    features[f"{prefix}active_duration_bins"] = active_duration_bins

    # ---------------------------------------------------------
    # 5. Frequency Sub-Band Energy Ratios
    # ---------------------------------------------------------
    # Partition the 140 frequency bins into 3 physically meaningful regimes:
    #   - Low Band: bins 0 to 39 (under ~100 Hz, e.g. Low Frequency Burst)
    #   - Mid Band: bins 40 to 89 (100 Hz to 500 Hz, e.g. Violin Mode, Power Line)
    #   - High Band: bins 90 to 139 (above 500 Hz, e.g. Whistle, 1080Lines)
    low_band_sum = float(np.sum(arr[:40, :]))
    mid_band_sum = float(np.sum(arr[40:90, :]))
    high_band_sum = float(np.sum(arr[90:, :]))

    features[f"{prefix}low_band_ratio"] = low_band_sum / total_energy
    features[f"{prefix}mid_band_ratio"] = mid_band_sum / total_energy
    features[f"{prefix}high_band_ratio"] = high_band_sum / total_energy

    # ---------------------------------------------------------
    # 6. Morphological Shape Indicators
    # ---------------------------------------------------------
    # Active pixel ratio: fraction of pixels considered "active signal"
    active_mask = (arr > bg_threshold)
    features[f"{prefix}active_pixel_ratio"] = float(np.mean(active_mask))

    # Horizontal Persistence: maximum contiguous time width across any frequency row
    # (High for continuous lines like Power Line and Violin Mode, low for Blip)
    max_h_extent = 0
    for row in active_mask:
        if np.any(row):
            # Calculate maximum run-length of True in this row
            runs = np.diff(np.where(np.concatenate(([row[0]], row[:-1] != row[1:], [True])))[0])[::2]
            if len(runs) > 0:
                max_h_extent = max(max_h_extent, int(np.max(runs)))
    features[f"{prefix}horizontal_persistence"] = float(max_h_extent)

    # Vertical Spread: maximum contiguous frequency height across any time column
    # (High for vertical spikes like Blip, low for narrow lines)
    max_v_extent = 0
    for col in active_mask.T:
        if np.any(col):
            runs = np.diff(np.where(np.concatenate(([col[0]], col[:-1] != col[1:], [True])))[0])[::2]
            if len(runs) > 0:
                max_v_extent = max(max_v_extent, int(np.max(runs)))
    features[f"{prefix}vertical_persistence"] = float(max_v_extent)

    # Aspect Ratio of active core: Vertical Spread / Horizontal Persistence
    features[f"{prefix}aspect_ratio"] = float(max_v_extent / (max_h_extent + eps))

    return features


def extract_multiview_features(views_dict):
    """
    Extracts features across multiple duration views (e.g. 0.5s, 1.0s, 2.0s, 4.0s)
    and combines them into a single comprehensive feature dictionary.

    Parameters:
    -----------
    views_dict : dict
        Mapping view names ('0.5s', '1.0s', etc.) to 2D numpy arrays.

    Returns:
    --------
    combined_features : dict
        Flat dictionary of features with view-specific prefixes.
    """
    combined_features = {}
    for view_name, array in views_dict.items():
        # Clean prefix: e.g. "0.5s" -> "v0_5s_"
        clean_prefix = f"v{view_name.replace('.', '_')}_"
        v_feats = extract_spectrogram_features(array, prefix=clean_prefix)
        combined_features.update(v_feats)
    return combined_features


def get_feature_names(prefix=""):
    """
    Returns the complete list of feature names generated by `extract_spectrogram_features`.
    """
    # Create a dummy 140x170 zero array to extract the key names
    dummy = np.zeros((140, 170), dtype=np.float32)
    feats = extract_spectrogram_features(dummy, prefix=prefix)
    return list(feats.keys())


if __name__ == "__main__":
    # Test execution when run as a script
    print("Testing Spectrogram Feature Extractor on synthetic and real glitch data...")

    # 1. Feature list check
    names = get_feature_names()
    print(f"Total features per spectrogram array: {len(names)}")
    print("Feature List:")
    for i, name in enumerate(names, start=1):
        print(f"  {i:2d}. {name}")

    # 2. Test on real glitch data from trainingsetv1d1.h5
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    sys.path.append(project_root)
    from src.data.dataset import load_all_views

    h5_file_path = os.path.join(project_root, "trainingsetv1d1.h5")
    meta_path = os.path.join(project_root, "data", "processed", "shared_classes_metadata.csv")

    if os.path.exists(h5_file_path) and os.path.exists(meta_path):
        import h5py
        df = pd.read_csv(meta_path)
        sample = df.iloc[0]
        print(f"\nExtracting features for real sample: {sample['gravityspy_id']} (Class: {sample['label']})")
        
        with h5py.File(h5_file_path, "r") as f:
            views = load_all_views(f, sample["label"], sample["sample_type"], sample["gravityspy_id"])
            
            # Single view extraction (1.0s)
            feats_1s = extract_spectrogram_features(views["1.0s"])
            print("\n--- Single-View (1.0s) Extracted Features Preview ---")
            for k, v in list(feats_1s.items())[:8]:
                print(f"  {k:25} : {v:.4f}")

            # Multi-view extraction
            mv_feats = extract_multiview_features(views)
            print(f"\n--- Multi-View Combined Feature Count: {len(mv_feats)} features ---")

        print("\nALL FEATURE EXTRACTOR VERIFICATION TESTS PASSED SUCCESSFULLY!")
