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
This project classifies transient noise artifacts ("glitches") in LIGO gravitational-wave detectors using the Gravity Spy public dataset (Zenodo record `1486046`). We systematically compare **interpretable feature-engineered models (Decision Trees, Random Forests)**, an **adapted reproduction of the Gravity Spy Convolutional Neural Network (CNN)**, **modern ImageNet-pretrained Vision Transfer Learning backbones (`ResNet-18`, `ConvNeXt-Tiny`, `ViT-B/16`) using Multi-View RGB `[0.5s, 1.0s, 2.0s]` spectrograms**, and **Vision-Language Models (`OpenAI CLIP ViT-B/16` via Zero-Shot Physics Prompting, Frozen Embedding Probes, and Prompt-Initialized Fine-Tuning)**, while evaluating both zero-shot and few-shot bidirectional domain adaptation across physical detector sites (**Option B: Cross-Detector Testing** between Hanford `H1` and Livingston `L1`).

## 2. Core Methodological Pillars
1. **Option B (Cross-Detector Transfer & Few-Shot Adaptation):** Filtered the dataset to **14 shared glitch classes** ($\ge 25$ samples at both `H1` and `L1`, retaining $6,340$ glitches / $79.59\%$ of the dataset) and evaluated zero-shot transfer and target-site few-shot adaptation (`0%, 5%, 10%, 20%` vs `100%` Native) in both directions ($H1 \leftrightarrow L1$) across **5 random seeds (`42–46`)**.
2. **Leak-Free Grouped Splitting:** Strictly partitioned glitches by `gravityspy_id` ($70\%$ Train / $15\%$ Val / $15\%$ Test) so all four duration zoom views ($0.5\text{s}, 1.0\text{s}, 2.0\text{s}, 4.0\text{s}$) remain in the exact same partition.
3. **Multi-Paradigm Representation Comparison:** Controlled comparison across **34 physically motivated features** ($20$ spectrogram + $14$ Omicron metadata), **scratch CNN**, **ImageNet-1K pretrained vision models (`ResNet-18`, `ConvNeXt-Tiny`, `ViT-B/16`)**, and **WIT-400M Vision-Language pretraining (`OpenAI CLIP ViT-B/16`)**.
4. **Rigorous Evaluation, Ablations & 4-Way Failure Analysis:** Reported unweighted **Macro-F1**, per-class F1 scores, Top-3 VLM recall, Permutation & MDI feature importances, 7 systematic feature ablations, and 4-way model complementarity (`RF` vs. `4-Layer CNN` vs. `ImageNet ViT-B/16` vs. `CLIP ViT-B/16`).

---

## 3. Pooled Benchmark Summary (Unseen Test Set, $N = 951$, Seed 42)

| Model | Paradigm / Input | Test Accuracy (%) | **Test Macro-F1** | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | Constant (`Blip`) | $28.71\%$ | $0.0319$ | $0.1281$ |
| **Baseline 2: Raw-Pixel Decision Tree** | $23,800$ raw pixels (`1.0s`) | $82.75\%$ | $0.7315$ | $0.8394$ |
| **Model 1: Feature Decision Tree** | $34$ engineered features | $91.59\%$ | $0.8536$ | $0.9152$ |
| **Model 2: Feature Random Forest** | $34$ engineered features | $97.06\%$ | $0.9376$ | $0.9699$ |
| **Model 3: Adapted Gravity Spy CNN (1.0s)** | Scratch CNN `(1, 140, 170)` | $96.85\%$ | $0.9466$ | $0.9682$ |
| **Model 4: ResNet-18 (1.0s Single-View)** | ImageNet Transfer (`1.0s`) | $97.27\%$ | $0.9543$ | $0.9727$ |
| **Model 5: ResNet-18 (Multi-View RGB)** | ImageNet Transfer (`RGB`) | $97.16\%$ | $0.9512$ | $0.9716$ |
| **Model 6: ConvNeXt-Tiny (Multi-View RGB)** | ImageNet Transfer (`RGB`) | $97.69\%$ | $0.9606$ | $0.9769$ |
| **Model 7: ViT-B/16 (Multi-View RGB)** | ImageNet Transfer (`RGB`) | **$97.79\%$** | **$0.9608$** | **$0.9783$** |
| **Model 8a: CLIP ViT-B/16 Zero-Shot (Basic)** | VLM Text Only (`0` train imgs) | $6.94\%$ | $0.0134$ | $0.0117$ |
| **Model 8b: CLIP ViT-B/16 Zero-Shot (Physics)** | VLM Physics Prompts (`Top-3=64.5%`) | $26.08\%$ | $0.1388$ | $0.2355$ |
| **Model 8c: CLIP ViT-B/16 Zero-Shot (Viridis)** | VLM Physics Prompts (`1.0s` Viridis) | $18.40\%$ | $0.1025$ | $0.1259$ |
| **Model 9: CLIP ViT-B/16 Frozen Linear Probe** | Frozen VLM + LogReg (`7.2K` params) | $94.32\%$ | $0.9060$ | $0.9431$ |
| **Model 10: CLIP ViT-B/16 Frozen MLP Probe** | Frozen VLM + 2-Layer MLP (`9.4s`) | $95.37\%$ | $0.9271$ | $0.9537$ |
| **Model 11: CLIP ViT-B/16 Fine-Tuned (Text-Init)** | Prompt-Init VLM Fine-Tune (`RGB`) | $96.95\%$ | $0.9474$ | $0.9697$ |

*Key Pooled Breakthroughs:*
- **ImageNet vs. CLIP Showdown (`ViT-B/16`):** Both ImageNet `ViT-B/16` (`0.9608`) and Text-Init `CLIP ViT-B/16` (`0.9474`) beat both RF (`0.9376`) and the 4-layer scratch CNN (`0.9466`). ImageNet pretraining outperforms CLIP by `+0.0134` on fine low-SNR pixel textures (`No_Glitch`, `Repeating_Blips`), while physics prompt engineering boosts zero-shot CLIP **`10.4×`** (`0.0134 → 0.1388`, **`64.46%` Top-3 accuracy** with `0` training images!).
- **4-Model Complementarity Ceiling:** Combining `RF` + `ImageNet ViT-B/16` + `CLIP ViT-B/16` yields a **`99.37%` (`945/951`) Oracle Fusion Ceiling**, leaving only **`6` unanimously label-ambiguous glitches (`0.63%`)**.

## 4. Option B: Bidirectional Cross-Detector Transfer & Few-Shot Recovery (5-Seed Mean ± Std)

### 4A. Zero-Shot Cross-Detector Transfer (`0%` Target-Site Labels)
| Model | `H1 → H1` (Within) | `H1 → L1` (Zero-Shot) | `L1 → L1` (Within) | `L1 → H1` (Zero-Shot) |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Random Forest (34 feats)** | $0.9329 \pm 0.0125$ | $0.6355 \pm 0.0147$ | $0.8970 \pm 0.0321$ | $0.7101 \pm 0.0101$ |
| **Frozen CLIP ViT-B/16 Linear Probe** | $0.8709 \pm 0.0183$ | $0.6806 \pm 0.0289$ | $0.8988 \pm 0.0236$ | $0.7063 \pm 0.0229$ |
| **Adapted Gravity Spy CNN (1.0s)** | $0.9011 \pm 0.0258$ | $0.7357 \pm 0.0233$ | $0.9151 \pm 0.0150$ | $0.7095 \pm 0.0347$ |
| **Pretrained ResNet-18 (Multi-View RGB)** | $0.9306 \pm 0.0322$ | **$0.8512 \pm 0.0322$** | **$0.9509 \pm 0.0117$** | **$0.7805 \pm 0.0276$** |

### 4B. Few-Shot Target-Detector Domain Adaptation (`H1 ↔ L1`, 5 Seeds)
| Adaptation Stage | Target Glitches (`L1` / `H1`) | **Step 15 Fine-Tuned ResNet-18 (`H1→L1` / `L1→H1`)** | **Step 16 Frozen CLIP Probe (`H1→L1` / `L1→H1`)** |
| :--- | :---: | :---: | :---: |
| **`0%` (Source Only)** | `0` / `0` | $0.8512$ / $0.7805$ | $0.6806$ / $0.7063$ |
| **`5%` Target Data** | `88` / `134` | $0.8929$ / $0.8342$ | $0.7589$ / $0.7736$ |
| **`10%` Target Data** | `176` / `268` | $0.9168$ / $0.8609$ | $0.8082$ / $0.7921$ |
| **`20%` Target Data** | `351` / `536` | **$0.9426$ / $0.8742$** | $0.8148$ / $0.7748$ |
| **`100%` Native Target** | `1,755` / `2,681` | $0.9509$ / $0.9306$ | $0.8988$ / $0.8709$ |

*Key Cross-Detector Discovery:* Feature RF collapses on `H1 → L1` (`0.6355`) due to `Violin_Mode` harmonic frequency shift (`0.9951 → 0.0637`). Even a **completely frozen** CLIP linear probe beats RF zero-shot (`0.6806`) and hits `0.8148` with `20%` target labels in seconds on CPU, while fine-tuning `ResNet-18 (Multi-View RGB)` jumps to **`0.8512`** zero-shot and recovers **`91.7%`** of the entire `H1 → L1` domain gap with just **`20%`** (`351`) Livingston glitches (`0.9426` vs. `0.9509` native)!

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
│   ├── models/      (baselines.py, tree_models.py, cnn.py, transfer_models.py, vlm_models.py)
│   ├── evaluation/  (metrics.py, plots.py, failure_analysis.py)
│   └── experiments/ (run_baselines.py, run_tree_models.py, run_ablations.py, run_cnn.py, run_cross_detector.py, run_transfer_learning.py, run_vlm_clip.py)
├── notebooks/
│   ├── 01_data_and_detector_eda.ipynb          # Step 1: EDA & H1 vs L1 distribution analysis
│   ├── 02_feature_verification.ipynb           # Step 4: Multi-view HDF5 & array verification
│   ├── 03_baseline_and_tree_models.ipynb       # Steps 7-9: Baselines, Decision Tree & Random Forest
│   ├── 04_cnn_training_colab.ipynb             # Steps 11-12: Adapted Gravity Spy CNN vs. Features
│   ├── 05_cross_detector_evaluation.ipynb      # Step 13: Option B Cross-Detector Transfer (5 seeds)
│   ├── 06_ablations_and_failure_analysis.ipynb # Steps 10 & 14: Feature Ablations & Visual Failure Analysis
│   ├── 07_transfer_learning_colab.ipynb        # Step 15: Pretrained Vision Transfer & Few-Shot Adaptation
│   └── 08_vlm_clip_colab.ipynb                 # Step 16: Vision-Language Models (OpenAI CLIP ViT-B/16)
├── outputs/
│   ├── figures/     # 18 publication-grade 300-DPI PNG visualizations
│   ├── models/      # rf_best_model.joblib, clip_embeddings_frozen.npz (5.7 MB), clip_embeddings_finetuned.npz (5.7 MB)
│   └── tables/      # All benchmark, tuning, ablation, cross-detector, transfer, VLM, and disagreement CSVs
└── report/          # Final research paper & presentation assets
```
