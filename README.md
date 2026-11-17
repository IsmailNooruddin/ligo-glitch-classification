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
This project classifies transient noise artifacts ("glitches") in LIGO gravitational-wave detectors using the Gravity Spy public dataset (Zenodo record `1486046`). We systematically compare an **adapted reproduction of the Gravity Spy Convolutional Neural Network (CNN)** against **interpretable feature-engineered models (Decision Trees, Random Forests)** and evaluate model transferability across physical detector sites (**Option B: Cross-Detector Testing** between Hanford `H1` and Livingston `L1`).

## 2. Core Methodological Pillars (Instructor Alignment)
1. **Option B (Cross-Detector Transfer):** Filtered the dataset to **14 shared glitch classes** ($\ge 25$ samples at both `H1` and `L1`, retaining $6,340$ glitches / $79.59\%$ of the dataset) for bidirectional evaluation ($H1 \leftrightarrow L1$).
2. **Leak-Free Grouped Splitting:** Strictly partitioned glitches by `gravityspy_id` ($70\%$ Train / $15\%$ Val / $15\%$ Test) so all four duration zoom views ($0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 4.0\text{s}$) remain in the exact same partition.
3. **Spectrogram-Supported Feature Engineering:** Extracted **34 physically motivated features** ($20$ directly from $140 \times 170$ spectrogram arrays + $14$ from Omicron hardware trigger metadata).
4. **Rigorous Evaluation & Ablations:** Reported unweighted **Macro-F1**, per-class F1 scores, Permutation & MDI feature importances, and a 7-condition systematic feature ablation study.

---

## 3. Interim Demo Benchmark Summary (Unseen Test Set, $N = 951$)

| Model | Input Features | Test Accuracy (%) | Test Macro-F1 | Weighted-F1 | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | — | $28.71\%$ | $0.0319$ | $0.1281$ | $<0.001\text{ s}$ |
| **Baseline 2: Raw-Pixel Decision Tree** | $23,800$ raw pixels | $82.75\%$ | $0.7315$ | $0.8394$ | $340.20\text{ s}$ |
| **Model 1: Feature Decision Tree** | $34$ engineered features | $91.59\%$ | $0.8536$ | $0.9152$ | $0.06\text{ s}$ |
| **Model 2: Feature Random Forest** | $34$ engineered features | **$97.06\%$** | **$0.9376$** | **$0.9699$** | $0.82\text{ s}$ |

---

## 4. Repository Structure

```text
AI Project/
├── data/
│   ├── README.md
│   └── processed/
│       ├── shared_classes_metadata.csv     # 6,340 glitches across 14 shared classes
│       ├── split_indices.json              # Leak-free grouped splits (Pooled, H1, L1, 5 seeds)
│       └── engineered_features.csv         # 6,340 x 34 pre-computed feature cache
├── src/
│   ├── data/
│   │   ├── class_selector.py               # Step 2: Shared class filtering
│   │   ├── split_utils.py                  # Step 3: Leak-free grouped splitting
│   │   └── dataset.py                      # Step 4: HDF5 loader & PyTorch Dataset
│   ├── features/
│   │   ├── extractor.py                    # Step 5: 20 spectrogram array features
│   │   └── metadata_features.py            # Step 6: 14 Omicron features & batch caching
│   ├── models/
│   │   ├── baselines.py                    # Steps 7-8: Majority & Raw-Pixel Tree baselines
│   │   ├── tree_models.py                  # Step 9: Feature Decision Tree & Random Forest
│   │   └── cnn.py                          # Step 11: Adapted 4-layer Gravity Spy CNN (Final Phase)
│   ├── evaluation/
│   │   ├── metrics.py                      # Macro-F1, Per-Class F1, Confusion Matrix, Seed stats
│   │   ├── plots.py                        # 300-DPI figure generation utilities
│   │   └── failure_analysis.py             # Step 14: Look-alike glitch failure analysis (Final Phase)
│   └── experiments/
│       ├── run_baselines.py                # Executes Steps 7 & 8
│       ├── run_tree_models.py              # Executes Step 9
│       ├── run_ablations.py                # Executes Step 10
│       ├── run_cnn.py                      # Executes Step 12 (Final Phase)
│       └── run_cross_detector.py           # Executes Step 13 (Final Phase)
├── notebooks/
│   ├── 01_data_and_detector_eda.ipynb      # Step 1: EDA & H1 vs L1 distribution analysis
│   ├── 02_feature_verification.ipynb       # Step 4: Multi-view HDF5 & array feature verification
│   ├── 03_baseline_and_tree_models.ipynb   # Steps 7-9: Baselines, Decision Tree & Random Forest
│   ├── 04_cnn_training_colab.ipynb         # Step 12: CNN training notebook (Final Phase)
│   ├── 05_cross_detector_evaluation.ipynb  # Step 13: H1 <-> L1 transfer evaluation (Final Phase)
│   └── 06_ablations_and_failure_analysis.ipynb # Step 10: Feature importances & systematic ablations
├── outputs/
│   ├── figures/                            # High-resolution 300-DPI PNG visualizations
│   ├── models/                             # Saved model artifacts (rf_best_model.joblib)
│   └── tables/                             # Benchmark, per-class, tuning, and ablation CSVs
└── report/                                 # Final research paper & presentation assets
```
