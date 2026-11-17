"""
Module: metrics.py
Purpose: Comprehensive evaluation metrics focusing on class imbalance and multi-seed stability.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Because LIGO glitches have severe class imbalance (e.g. 1,821 Blips vs 60 Chirps), standard
accuracy is dangerously misleading. In accordance with the instructor's explicit feedback, this
module calculates:
1. Macro-F1 (unweighted average across all classes)
2. Per-class Precision, Recall, and F1-score
3. Multi-seed aggregation (reporting mean ± standard deviation across runs)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


def compute_all_metrics(y_true, y_pred, class_names=None):
    """
    Computes a full suite of classification metrics.

    Parameters:
    -----------
    y_true : list or np.ndarray
        True class labels (strings or integers).
    y_pred : list or np.ndarray
        Predicted class labels.
    class_names : list of str, optional
        Alphabetical list of unique class names. If None, derived from y_true.

    Returns:
    --------
    metrics_dict : dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'macro_f1': Unweighted Macro-F1 score (primary course metric)
        - 'weighted_f1': Support-weighted F1 score
        - 'per_class': DataFrame with per-class Precision, Recall, F1, and Support
        - 'confusion_matrix': Raw 2D confusion matrix
        - 'class_names': List of classes corresponding to matrix rows/cols
    """
    # Determine unique classes if not provided
    if class_names is None:
        class_names = sorted(list(set(y_true).union(set(y_pred))))

    # 1. Global Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=class_names, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=class_names, average='weighted', zero_division=0)

    # 2. Per-Class Precision, Recall, and F1
    per_class_f1 = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)

    # Calculate support (true occurrences per class in test set)
    support_counts = []
    for cls in class_names:
        support_counts.append(int(np.sum(np.array(y_true) == cls)))

    per_class_df = pd.DataFrame({
        'Class': class_names,
        'Precision': [round(p, 4) for p in per_class_prec],
        'Recall': [round(r, 4) for r in per_class_rec],
        'F1-Score': [round(f, 4) for f in per_class_f1],
        'Support': support_counts
    })

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    return {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': per_class_df,
        'confusion_matrix': cm,
        'class_names': class_names
    }


def aggregate_seed_results(seed_results_list):
    """
    Computes the Mean and Standard Deviation of Macro-F1 and Accuracy across multiple random seeds.

    Parameters:
    -----------
    seed_results_list : list of dict
        List of metrics dictionaries produced by compute_all_metrics across seeds.

    Returns:
    --------
    summary : dict
        Mean and standard deviation for key metrics.
    """
    macro_f1s = [res['macro_f1'] for res in seed_results_list]
    accuracies = [res['accuracy'] for res in seed_results_list]

    return {
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'num_seeds': len(seed_results_list)
    }


def print_metrics_summary(model_name, metrics):
    """
    Prints a clean, formatted terminal summary of model performance.
    """
    print(f"\n==================================================================")
    print(f" EVALUATION SUMMARY: {model_name}")
    print(f"==================================================================")
    print(f" Overall Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f" Macro-F1 Score   : {metrics['macro_f1']:.4f}  <-- PRIMARY METRIC")
    print(f" Weighted-F1 Score: {metrics['weighted_f1']:.4f}")
    print(f"------------------------------------------------------------------")
    print(f" Per-Class Performance:")
    print(metrics['per_class'].to_string(index=False))
    print(f"==================================================================\n")
