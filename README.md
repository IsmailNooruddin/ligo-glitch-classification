# Classifying LIGO Glitches: Comparing CNNs and Simple Features

**Course:** CS/CE 351/363 - Artificial Intelligence (Fall 2026)  
**Institution:** Habib University  
**Group:** L4 Group 23  
**Group Members:**  
- Ismail Nooruddin (09958)  
- Munir Tariq (10067)  
- Ramsha Arshad (09278)  

---

## 1. Project Overview
This project classifies transient noise artifacts ("glitches") in LIGO gravitational-wave detectors using the Gravity Spy public dataset (Zenodo record `1486046`). We systematically compare an **adapted reproduction of the Gravity Spy Convolutional Neural Network (CNN)** against **interpretable feature-engineered models (Decision Trees, Random Forests)** and evaluate bidirectional model transferability across physical detector sites (**Option B: Cross-Detector Testing** between Hanford `H1` and Livingston `L1`).

## 2. Core Methodological Pillars
1. **Option B (Cross-Detector Transfer):** Filtered the dataset to **14 shared glitch classes** ($\ge 25$ samples at both `H1` and `L1`, retaining $6,340$ glitches / $79.59\%$ of the dataset) and evaluated transfer in both directions ($H1 \leftrightarrow L1$) across **5 random seeds (`42–46`)**.
2. **Leak-Free Grouped Splitting:** Strictly partitioned glitches by `gravityspy_id` ($70\%$ Train / $15\%$ Val / $15\%$ Test) so all four duration zoom views ($0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 4.0\text{s}$) remain in the exact same partition.
3. **Spectrogram-Supported Feature Engineering:** Extracted **34 physically motivated features** ($20$ directly from $140 \times 170$ spectrogram arrays + $14$ from Omicron hardware trigger metadata).
4. **Rigorous Evaluation, Ablations & Visual Failure Analysis:** Reported unweighted **Macro-F1**, per-class F1 scores, Permutation & MDI feature importances, 7 systematic feature ablations, and RF vs. CNN visual disagreement grids.

---

## 3. Pooled Benchmark Summary (Unseen Test Set, $N = 951$, Seed 42)

| Model | Input Features | Test Accuracy (%) | Test Macro-F1 | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | — | $28.71\%$ | $0.0319$ | $0.1281$ |
| **Baseline 2: Raw-Pixel Decision Tree** | $23,800$ raw pixels | $82.75\%$ | $0.7315$ | $0.8394$ |
| **Model 1: Feature Decision Tree** | $34$ engineered features | $91.59\%$ | $0.8536$ | $0.9152$ |
| **Model 2: Feature Random Forest** | $34$ engineered features | **$97.06\%$** | $0.9376$ | **$0.9699$** |
| **Model 3: Adapted Gravity Spy CNN (1.0s)** | `(1, 140, 170)` spectrogram | $96.85\%$ | **$0.9466$** | $0.9682$ |

## 4. Option B: Bidirectional Cross-Detector Transfer (5-Seed Mean ± Std)

| Model | `H1 → H1` (Within) | `H1 → L1` (Transfer) | `L1 → L1` (Within) | `L1 → H1` (Transfer) |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Random Forest (34 feats)** | $0.9329 \pm 0.0125$ | $0.6355 \pm 0.0147$ (`-0.2615`) | $0.8970 \pm 0.0321$ | $0.7101 \pm 0.0101$ (`-0.2228`) |
| **Adapted Gravity Spy CNN (1.0s)** | $0.9011 \pm 0.0258$ | **$0.7357 \pm 0.0233$** (`-0.1794`) | $0.9151 \pm 0.0150$ | $0.7095 \pm 0.0347$ (`-0.1916`) |

*Key Discovery:* Random Forest loses **`-0.2615`** Macro-F1 on `H1 → L1` largely because `Violin_Mode` harmonic frequencies shift slightly between Hanford and Livingston (`Violin_Mode` F1 crashes `0.9951 → 0.0637`), whereas our translation-equivariant CNN preserves `Violin_Mode` (`0.9219` F1) and beats RF by **`+0.1002` Macro-F1** across all 5 seeds!

---

## 5. Repository Structure

```text
AI Project/
├── data/processed/
│   ├── shared_classes_metadata.csv             # 6,340 glitches across 14 shared classes
│   ├── split_indices.json                      # Leak-free grouped splits (Pooled, H1, L1, 5 seeds)
│   └── engineered_features.csv                 # 6,340 x 34 pre-computed feature cache
├── src/
│   ├── data/        (class_selector.py, split_utils.py, dataset.py)
│   ├── features/    (extractor.py, metadata_features.py)
│   ├── models/      (baselines.py, tree_models.py, cnn.py)
│   ├── evaluation/  (metrics.py, plots.py, failure_analysis.py)
│   └── experiments/ (run_baselines.py, run_tree_models.py, run_ablations.py, run_cnn.py, run_cross_detector.py)
├── notebooks/
│   ├── 01_data_and_detector_eda.ipynb          # Step 1: EDA & H1 vs L1 distribution analysis
│   ├── 02_feature_verification.ipynb           # Step 4: Multi-view HDF5 & array verification
│   ├── 03_baseline_and_tree_models.ipynb       # Steps 7-9: Baselines, Decision Tree & Random Forest
│   ├── 04_cnn_training_colab.ipynb             # Steps 11-12: Adapted Gravity Spy CNN vs. Features
│   ├── 05_cross_detector_evaluation.ipynb      # Step 13: Option B Cross-Detector Transfer (5 seeds)
│   └── 06_ablations_and_failure_analysis.ipynb # Steps 10 & 14: Feature Ablations & Visual Failure Analysis
├── outputs/
│   ├── figures/     # 12 publication-grade 300-DPI PNG visualizations
│   ├── models/      # rf_best_model.joblib & cnn_best_checkpoint.pt
│   └── tables/      # All benchmark, tuning, ablation, cross-detector, and disagreement CSVs
└── report/          # Final research paper & presentation assets
```
