"""
Module: plots.py
Purpose: Publication-quality visualization utilities for LIGO glitch classification.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Generates and saves 300-DPI figures:
1. Normalized Confusion Matrix Heatmaps (`plot_confusion_matrix`)
2. Dual-Panel Feature Importance Bar Charts (`plot_feature_importances`)
3. Systematic Ablation Impact Bar Charts (`plot_ablation_impact`)
4. Per-Class F1 Progression Grouped Bar Charts (`plot_per_class_f1_comparison`)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plots and optionally saves a row-normalized confusion matrix heatmap.

    Parameters:
    -----------
    cm : np.ndarray, shape (C, C)
        Raw 2D confusion matrix.
    class_names : list of str
        Ordered list of class labels corresponding to matrix rows/columns.
    title : str
        Title displayed at the top of the heatmap.
    save_path : str, optional
        File path to save the rendered PNG figure (300 DPI).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Recall'},
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Predicted Glitch Class", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Glitch Class", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix figure saved to: {save_path}")

    plt.close(fig)


def plot_feature_importances(importance_df, top_n=15, save_path=None):
    """
    Plots a dual-panel horizontal bar chart comparing Permutation Importance vs MDI Gini Importance.

    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame containing 'Feature', 'Permutation_Mean', 'Permutation_Std', and 'MDI_Gini'.
    top_n : int, default=15
        Number of top features to display in each panel.
    save_path : str, optional
        File path to save the rendered PNG figure (300 DPI).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=False)

    # Panel A: Permutation Importance
    top_perm = importance_df.sort_values(by='Permutation_Mean', ascending=False).head(top_n).iloc[::-1]
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

    # Panel B: MDI Gini Importance
    top_mdi = importance_df.sort_values(by='MDI_Gini', ascending=False).head(top_n).iloc[::-1]
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

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Feature importance figure saved to: {save_path}")

    plt.close(fig)


def plot_ablation_impact(ablation_df, baseline_macro_f1=0.9376, save_path=None):
    """
    Plots horizontal bar chart showing Macro-F1 across feature ablation experiments.

    Parameters:
    -----------
    ablation_df : pd.DataFrame
        DataFrame with 'Experiment', 'Macro-F1', and 'Delta_Macro_F1'.
    baseline_macro_f1 : float
        Reference full-model Macro-F1 score.
    save_path : str, optional
        File path to save the rendered PNG figure (300 DPI).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#1f77b4' if d == 0 else ('#d62728' if d < -0.05 else '#ff7f0e') for d in ablation_df['Delta_Macro_F1']]
    bars = ax.barh(
        ablation_df['Experiment'].iloc[::-1],
        ablation_df['Macro-F1'].iloc[::-1],
        color=colors[::-1],
        edgecolor='black',
        alpha=0.85
    )

    ax.axvline(x=baseline_macro_f1, color='#1f77b4', linestyle='--', linewidth=1.5, label=f"Full Model Macro-F1 ({baseline_macro_f1:.4f})")
    ax.set_xlim(0.70, 0.98)
    ax.set_xlabel("Test Set Macro-F1 Score", fontsize=11, fontweight='bold')
    ax.set_title("Impact of Feature-Group Ablations on LIGO Glitch Classification (Random Forest)", fontsize=12, fontweight='bold', pad=12)

    for bar, delta in zip(bars, ablation_df['Delta_Macro_F1'].iloc[::-1]):
        w = bar.get_width()
        txt = f" {w:.4f} (Ref)" if delta == 0 else f" {w:.4f} ({delta:+.4f})"
        ax.text(w + 0.003, bar.get_y() + bar.get_height() / 2, txt, va='center', fontsize=9, fontweight='bold')

    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Ablation impact figure saved to: {save_path}")

    plt.close(fig)


def plot_per_class_f1_comparison(per_class_df, save_path=None):
    """
    Plots a grouped horizontal bar chart comparing per-class F1 scores across
    Raw-Pixel Decision Tree, Feature Decision Tree, and Feature Random Forest.
    """
    df_sorted = per_class_df.sort_values(by='Random_Forest_F1', ascending=True).reset_index(drop=True)
    y = np.arange(len(df_sorted))
    height = 0.26

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.barh(y - height, df_sorted['Raw_Pixel_Tree_F1'], height, label='Baseline 2: Raw-Pixel Tree (23,800 dims)', color='#d62728', alpha=0.82, edgecolor='black')
    ax.barh(y, df_sorted['Feature_Tree_F1'], height, label='Model 1: Feature Decision Tree (34 dims)', color='#ff7f0e', alpha=0.85, edgecolor='black')
    ax.barh(y + height, df_sorted['Random_Forest_F1'], height, label='Model 2: Feature Random Forest (34 dims)', color='#2ca02c', alpha=0.88, edgecolor='black')

    ax.set_yticks(y)
    ax.set_yticklabels(df_sorted['Class'], fontsize=10, fontweight='bold')
    ax.set_xlim(0.0, 1.08)
    ax.set_xlabel("Test Set F1-Score", fontsize=11, fontweight='bold')
    ax.set_title("Per-Class F1-Score Progression: Raw Pixels vs. Engineered Feature Models", fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10, frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Per-class F1 progression figure saved to: {save_path}")

    plt.close(fig)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    per_class_csv = os.path.join(project_root, "outputs", "tables", "per_class_f1_summary.csv")
    if os.path.exists(per_class_csv):
        df_pc = pd.read_csv(per_class_csv)
        out_fig = os.path.join(project_root, "outputs", "figures", "per_class_f1_progression.png")
        plot_per_class_f1_comparison(df_pc, save_path=out_fig)
