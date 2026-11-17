"""
Module: tree_models.py
Purpose: Implement, tune, and evaluate Decision Tree and Random Forest classifiers on 34 engineered features.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Instead of treating spectrograms as 23,800 unconstrained pixel numbers (which leads to extreme noise
overfitting and takes minutes to train), this module trains interpretable tree-based models on our 34
physically motivated engineered features (20 spectrogram features + 14 Omicron metadata features).
Hyperparameters (max_depth, n_estimators, class_weight) are tuned on the validation set using Macro-F1
to combat severe class imbalance before final test set evaluation.
"""

import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score


# Exact 34 feature column names defined in src/features/extractor.py & metadata_features.py
SPECTROGRAM_FEATURE_COLS = [
    'spec_peak_energy',
    'spec_mean_energy',
    'spec_std_energy',
    'spec_energy_contrast',
    'spec_energy_concentration',
    'spec_peak_freq_bin',
    'spec_peak_time_bin',
    'spec_spectral_centroid',
    'spec_spectral_spread',
    'spec_spectral_skewness',
    'spec_temporal_centroid',
    'spec_temporal_spread',
    'spec_active_duration_bins',
    'spec_low_band_ratio',
    'spec_mid_band_ratio',
    'spec_high_band_ratio',
    'spec_active_pixel_ratio',
    'spec_horizontal_persistence',
    'spec_vertical_persistence',
    'spec_aspect_ratio'
]

METADATA_FEATURE_COLS = [
    'meta_duration',
    'meta_peak_frequency',
    'meta_central_freq',
    'meta_bandwidth',
    'meta_amplitude',
    'meta_snr',
    'meta_log_duration',
    'meta_log_peak_frequency',
    'meta_log_central_freq',
    'meta_log_bandwidth',
    'meta_log_snr',
    'meta_log_amplitude',
    'meta_q_factor_approx',
    'meta_bandwidth_ratio'
]

ALL_FEATURE_COLS = SPECTROGRAM_FEATURE_COLS + METADATA_FEATURE_COLS


def get_feature_names(feature_set='all'):
    """
    Returns list of feature column names for a given feature set.

    Parameters:
    -----------
    feature_set : str, default='all'
        One of 'all', 'spectrogram', or 'metadata'.
    """
    if feature_set == 'all':
        return list(ALL_FEATURE_COLS)
    elif feature_set == 'spectrogram':
        return list(SPECTROGRAM_FEATURE_COLS)
    elif feature_set == 'metadata':
        return list(METADATA_FEATURE_COLS)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Choose 'all', 'spectrogram', or 'metadata'.")


def prepare_feature_matrices(train_df, val_df, test_df, feature_cols=None):
    """
    Extracts numpy feature matrices and target arrays from partition DataFrames.

    Parameters:
    -----------
    train_df, val_df, test_df : pd.DataFrame
        DataFrames containing engineered features and 'label'.
    feature_cols : list of str, optional
        Subset of features to extract. Defaults to ALL_FEATURE_COLS (34 features).

    Returns:
    --------
    (X_train, y_train), (X_val, y_val), (X_test, y_test) : tuples of np.ndarray
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['label'].values

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['label'].values

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df['label'].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, class_weight='balanced', random_state=42):
    """
    Trains a single feature-based Decision Tree classifier.

    Parameters:
    -----------
    X_train : np.ndarray, shape (N, D)
        Training feature matrix.
    y_train : np.ndarray, shape (N,)
        Target label array.
    max_depth : int or None, default=None
        Maximum tree depth. None allows nodes to expand until leaves are pure.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum samples required to be at a leaf node.
    class_weight : str or None, default='balanced'
        'balanced' weights inversely proportional to class frequencies to combat imbalance.
    random_state : int, default=42
        Reproducibility seed.

    Returns:
    --------
    model : DecisionTreeClassifier
        Fitted decision tree.
    """
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def tune_decision_tree(X_train, y_train, X_val, y_val, class_names,
                       depth_options=None, class_weight_options=None,
                       min_samples_split_options=None, random_state=42):
    """
    Tunes Decision Tree hyperparameters using validation set Macro-F1.

    Returns:
    --------
    best_model : DecisionTreeClassifier
    best_params : dict
    results_df : pd.DataFrame
    """
    if depth_options is None:
        depth_options = [4, 6, 8, 10, 12, 15, None]
    if class_weight_options is None:
        class_weight_options = [None, 'balanced']
    if min_samples_split_options is None:
        min_samples_split_options = [2, 5, 10]

    records = []
    best_score = -1.0
    best_model = None
    best_params = None

    print(f"Tuning Decision Tree across {len(depth_options) * len(class_weight_options) * len(min_samples_split_options)} configurations...")

    for depth in depth_options:
        for cw in class_weight_options:
            for split in min_samples_split_options:
                t0 = time.time()
                clf = DecisionTreeClassifier(
                    max_depth=depth,
                    class_weight=cw,
                    min_samples_split=split,
                    random_state=random_state
                )
                clf.fit(X_train, y_train)
                fit_time = time.time() - t0

                val_preds = clf.predict(X_val)
                val_macro = f1_score(y_val, val_preds, labels=class_names, average='macro', zero_division=0)
                val_acc = accuracy_score(y_val, val_preds)

                # Track candidate
                records.append({
                    'max_depth': str(depth),
                    'class_weight': str(cw),
                    'min_samples_split': split,
                    'val_macro_f1': round(val_macro, 4),
                    'val_accuracy': round(val_acc * 100, 2),
                    'train_time_sec': round(fit_time, 4)
                })

                if val_macro > best_score:
                    best_score = val_macro
                    best_model = clf
                    best_params = {
                        'max_depth': depth,
                        'class_weight': cw,
                        'min_samples_split': split,
                        'val_macro_f1': val_macro
                    }

    results_df = pd.DataFrame(records).sort_values(by='val_macro_f1', ascending=False).reset_index(drop=True)
    print(f"Optimal Decision Tree params: {best_params} (Val Macro-F1 = {best_score:.4f})")
    return best_model, best_params, results_df


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                        min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                        class_weight='balanced', random_state=42, n_jobs=-1):
    """
    Trains a feature-based Random Forest classifier.

    Parameters:
    -----------
    X_train : np.ndarray, shape (N, D)
    y_train : np.ndarray, shape (N,)
    n_estimators : int, default=100
        Number of decision trees in forest ensemble.
    max_depth : int or None, default=None
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : str or float, default='sqrt'
    class_weight : str or None, default='balanced'
    random_state : int, default=42
    n_jobs : int, default=-1
        Parallel threads (-1 uses all available CPU cores).

    Returns:
    --------
    model : RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, X_val, y_val, class_names,
                       n_estimators_options=None, depth_options=None,
                       class_weight_options=None, random_state=42, n_jobs=-1):
    """
    Tunes Random Forest hyperparameters using validation set Macro-F1.

    Returns:
    --------
    best_model : RandomForestClassifier
    best_params : dict
    results_df : pd.DataFrame
    """
    if n_estimators_options is None:
        n_estimators_options = [50, 100, 200]
    if depth_options is None:
        depth_options = [8, 12, 16, 20, None]
    if class_weight_options is None:
        class_weight_options = [None, 'balanced', 'balanced_subsample']

    records = []
    best_score = -1.0
    best_model = None
    best_params = None

    total_configs = len(n_estimators_options) * len(depth_options) * len(class_weight_options)
    print(f"Tuning Random Forest across {total_configs} configurations...")

    for n_est in n_estimators_options:
        for depth in depth_options:
            for cw in class_weight_options:
                t0 = time.time()
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    class_weight=cw,
                    max_features='sqrt',
                    random_state=random_state,
                    n_jobs=n_jobs
                )
                rf.fit(X_train, y_train)
                fit_time = time.time() - t0

                val_preds = rf.predict(X_val)
                val_macro = f1_score(y_val, val_preds, labels=class_names, average='macro', zero_division=0)
                val_acc = accuracy_score(y_val, val_preds)

                records.append({
                    'n_estimators': n_est,
                    'max_depth': str(depth),
                    'class_weight': str(cw),
                    'val_macro_f1': round(val_macro, 4),
                    'val_accuracy': round(val_acc * 100, 2),
                    'train_time_sec': round(fit_time, 2)
                })

                if val_macro > best_score:
                    best_score = val_macro
                    best_model = rf
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'class_weight': cw,
                        'val_macro_f1': val_macro
                    }

    results_df = pd.DataFrame(records).sort_values(by='val_macro_f1', ascending=False).reset_index(drop=True)
    print(f"Optimal Random Forest params: {best_params} (Val Macro-F1 = {best_score:.4f})")
    return best_model, best_params, results_df
