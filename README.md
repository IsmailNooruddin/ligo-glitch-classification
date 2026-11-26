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
This project classifies transient noise artifacts ("glitches") in LIGO gravitational-wave detectors using the Gravity Spy public dataset (Zenodo record `1486046`). We systematically compare **interpretable feature-engineered models (Decision Trees, Random Forests)**, an **adapted reproduction of the Gravity Spy Convolutional Neural Network (CNN)**, and **modern ImageNet-pretrained Vision Transfer Learning backbones (`ResNet-18`, `ConvNeXt-Tiny`, `ViT-B/16`) using Multi-View RGB `[0.5s, 1.0s, 2.0s]` spectrograms**, while evaluating both zero-shot and few-shot bidirectional domain adaptation across physical detector sites (**Option B: Cross-Detector Testing** between Hanford `H1` and Livingston `L1`).

## 2. Core Methodological Pillars
1. **Option B (Cross-Detector Transfer & Few-Shot Adaptation):** Filtered the dataset to **14 shared glitch classes** ($\ge 25$ samples at both `H1` and `L1`, retaining $6,340$ glitches / $79.59\%$ of the dataset) and evaluated both zero-shot transfer and target-site few-shot fine-tuning (`0%, 5%, 10%, 20%` vs `100%` Native) in both directions ($H1 \leftrightarrow L1$) across **5 random seeds (`42–46`)**.
2. **Leak-Free Grouped Splitting:** Strictly partitioned glitches by `gravityspy_id` ($70\%$ Train / $15\%$ Val / $15\%$ Test) so all four duration zoom views ($0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 4.0\text{s}$) remain in the exact same partition.
3. **Spectrogram-Supported Feature Engineering & Multi-View RGB Vision Transfer:** Extracted **34 physically motivated features** ($20$ directly from $140 \times 170$ spectrogram arrays + $14$ from Omicron hardware trigger metadata) and compared them against ImageNet-pretrained backbones fed Multi-View RGB `[0.5s, 1.0s, 2.0s]` spectrograms.
4. **Rigorous Evaluation, Ablations & Visual Failure Analysis:** Reported unweighted **Macro-F1**, per-class F1 scores, Permutation & MDI feature importances, 7 systematic feature ablations, and 3-way model disagreement analysis (`RF` vs. `4-Layer CNN` vs. `ViT-B/16`).

---

## 3. Pooled Benchmark Summary (Unseen Test Set, $N = 951$, Seed 42)

| Model | Input Representation | Test Accuracy (%) | **Test Macro-F1** | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | — | $28.71\%$ | $0.0319$ | $0.1281$ |
| **Baseline 2: Raw-Pixel Decision Tree** | $23,800$ raw pixels (`1.0s`) | $82.75\%$ | $0.7315$ | $0.8394$ |
| **Model 1: Feature Decision Tree** | $34$ engineered features | $91.59\%$ | $0.8536$ | $0.9152$ |
| **Model 2: Feature Random Forest** | $34$ engineered features | $97.06\%$ | $0.9376$ | $0.9699$ |
| **Model 3: Adapted Gravity Spy CNN (1.0s)** | `(1, 140, 170)` spectrogram | $96.85\%$ | $0.9466$ | $0.9682$ |
| **Model 4: ResNet-18 (1.0s Single-View)** | `(1, 140, 170) → RGB` | $97.27\%$ | $0.9543$ | $0.9727$ |
| **Model 5: ResNet-18 (Multi-View RGB)** | `[0.5s, 1.0s, 2.0s]` RGB | $97.16\%$ | $0.9512$ | $0.9716$ |
| **Model 6: ConvNeXt-Tiny (Multi-View RGB)** | `[0.5s, 1.0s, 2.0s]` RGB | $97.69\%$ | $0.9606$ | $0.9769$ |
| **Model 7: ViT-B/16 (Multi-View RGB)** | `[0.5s, 1.0s, 2.0s]` RGB | **$97.79\%$** | **$0.9608$** | **$0.9783$** |

*Key Pooled Breakthroughs:* Every ImageNet-pretrained backbone outperformed both Feature RF and the Adapted 4-Layer CNN. Mapping `[0.5s, 1.0s, 2.0s]` views to RGB pushed `Repeating_Blips` from `0.8684` (RF) and `0.9383` (4-layer CNN) to **`1.0000` F1** on both `ConvNeXt-Tiny` and `ViT-B/16`. Combining pure vision (`ViT-B/16`, `21/951` errors) with physics-metadata features (`RF`) yields a **`99.26%` (`944/951`) Oracle Fusion Ceiling**.

## 4. Option B: Bidirectional Cross-Detector Transfer & Few-Shot Recovery (5-Seed Mean ± Std)

### 4A. Zero-Shot Cross-Detector Transfer (`0%` Target-Site Labels)
| Model | `H1 → H1` (Within) | `H1 → L1` (Zero-Shot) | `L1 → L1` (Within) | `L1 → H1` (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Random Forest (34 feats)** | $0.9329 \pm 0.0125$ | $0.6355 \pm 0.0147$ | $0.8970 \pm 0.0321$ | $0.7101 \pm 0.0101$ |
| **Adapted Gravity Spy CNN (1.0s)** | $0.9011 \pm 0.0258$ | $0.7357 \pm 0.0233$ | $0.9151 \pm 0.0150$ | $0.7095 \pm 0.0347$ |
| **Pretrained ResNet-18 (Multi-View RGB)** | $0.9306 \pm 0.0322$ | **$0.8512 \pm 0.0322$** | **$0.9509 \pm 0.0117$** | **$0.7805 \pm 0.0276$** |

### 4B. Step 15 Novelty: Few-Shot Target-Detector Domain Adaptation (`ResNet-18 Multi-View RGB`, 5 Seeds)
| Adaptation Stage | `H1 → L1` Target Glitches | **`H1 → L1` Macro-F1** | `L1 → H1` Target Glitches | **`L1 → H1` Macro-F1** |
| :--- | :---: | :---: | :---: | :---: |
| **`0%` (Zero-Shot Transfer)** | `0` | $0.8512 \pm 0.0322$ | `0` | $0.7805 \pm 0.0276$ |
| **`5%` Target Fine-Tuning** | `88` | $0.8929 \pm 0.0229$ (`+0.0417`) | `134` | $0.8342 \pm 0.0282$ (`+0.0536`) |
| **`10%` Target Fine-Tuning** | `176` | $0.9168 \pm 0.0187$ (`+0.0656`) | `268` | $0.8609 \pm 0.0200$ (`+0.0804`) |
| **`20%` Target Fine-Tuning** | `351` | **$0.9426 \pm 0.0089$** (`+0.0914`) | `536` | **$0.8742 \pm 0.0348$** (`+0.0937`) |
| **`100%` Native Target Upper Bound** | `1,755` | $0.9509 \pm 0.0117$ | `2,681` | $0.9306 \pm 0.0322$ |

*Key Cross-Detector Discovery:* Feature RF collapses on `H1 → L1` (`0.6355`) largely because `Violin_Mode` harmonic frequencies shift between sites (`0.9951 → 0.0637`). Pretrained `ResNet-18 (Multi-View RGB)` jumps to **`0.8512`** zero-shot (`+0.2157` over RF, `+0.1155` over 4-layer CNN), surpasses both Step 13 native models with just **`10%`** Livingston data (`0.9168`), and recovers **`91.7%`** of the entire `H1 → L1` domain gap with just **`20%`** (`351`) Livingston glitches (`0.9426` vs. `0.9509` native)!

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
│   ├── models/      (baselines.py, tree_models.py, cnn.py, transfer_models.py)
│   ├── evaluation/  (metrics.py, plots.py, failure_analysis.py)
│   └── experiments/ (run_baselines.py, run_tree_models.py, run_ablations.py, run_cnn.py, run_cross_detector.py, run_transfer_learning.py)
├── notebooks/
│   ├── 01_data_and_detector_eda.ipynb          # Step 1: EDA & H1 vs L1 distribution analysis
│   ├── 02_feature_verification.ipynb           # Step 4: Multi-view HDF5 & array verification
│   ├── 03_baseline_and_tree_models.ipynb       # Steps 7-9: Baselines, Decision Tree & Random Forest
│   ├── 04_cnn_training_colab.ipynb             # Steps 11-12: Adapted Gravity Spy CNN vs. Features
│   ├── 05_cross_detector_evaluation.ipynb      # Step 13: Option B Cross-Detector Transfer (5 seeds)
│   ├── 06_ablations_and_failure_analysis.ipynb # Steps 10 & 14: Feature Ablations & Visual Failure Analysis
│   └── 07_transfer_learning_colab.ipynb        # Step 15: Pretrained Vision Transfer & Few-Shot Adaptation
├── outputs/
│   ├── figures/     # 15 publication-grade 300-DPI PNG visualizations
│   ├── models/      # rf_best_model.joblib, cnn_best_checkpoint.pt (local transfer_best_checkpoint.pt ignored by git)
│   └── tables/      # All benchmark, tuning, ablation, cross-detector, transfer, and disagreement CSVs
└── report/          # Final research paper & presentation assets
```
