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
This project classifies transient noise artifacts ("glitches") in LIGO gravitational-wave detectors using the Gravity Spy public dataset (Zenodo record `1486046`). We systematically benchmark **five distinct AI paradigms** on identical leak-free partitions:
1. **Classical Feature Engineering** (Decision Trees & Random Forests on `34` physical Omicron + spectrogram features),
2. **Scratch Deep Learning** (Adapted Gravity Spy 4-layer PyTorch CNN on `1.0s` spectrograms),
3. **Modern Vision Transfer Learning** (ImageNet-pretrained `ResNet-18`, `ConvNeXt-Tiny`, `ViT-B/16` with Multi-View RGB `[0.5s, 1.0s, 2.0s]`),
4. **Vision-Language Models** (`OpenAI CLIP ViT-B/16` via Zero-Shot Physics Prompting, Frozen Embedding Probes, and Prompt-Initialized Fine-Tuning), and
5. **Multimodal Physics–Vision Hybrid Fusion** (combining `34` physical Omicron/spectrogram features with frozen CLIP embeddings and calibrated ImageNet `ViT-B/16` predictions), alongside bidirectional **5-Seed Cross-Detector Transfer & Few-Shot Adaptation (`H1 ↔ L1`)**.

---

## 2. Grand Finale Pooled Leaderboard (Unseen Test Set, $N = 951$, Seed 42)

| Model | Paradigm / Input | Test Accuracy (%) | **Test Macro-F1** | Weighted-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** | Constant (`Blip`) | $28.71\%$ | $0.0319$ | $0.1281$ |
| **Baseline 2: Raw-Pixel Decision Tree** | $23,800$ raw pixels (`1.0s`) | $82.75\%$ | $0.7315$ | $0.8394$ |
| **Model 1: Feature Decision Tree** | $34$ physical features | $91.59\%$ | $0.8536$ | $0.9152$ |
| **Model 2: Feature Random Forest** | $34$ physical features | $97.06\%$ | $0.9376$ | $0.9699$ |
| **Model 3: Adapted Gravity Spy CNN (1.0s)** | Scratch CNN `(1, 140, 170)` | $96.85\%$ | $0.9466$ | $0.9682$ |
| **Model 4: CLIP Zero-Shot (Basic → Physics)** | VLM Text Only (`0` train imgs) | $6.94\% \to 26.08\%$ | $0.0134 \to 0.1388$ | $0.0117 \to 0.2355$ |
| **Model 5: CLIP Frozen MLP Probe** | Frozen VLM + MLP (`9.4s`) | $95.37\%$ | $0.9271$ | $0.9537$ |
| **Model 6: CLIP Fine-Tuned (Text-Init, RGB)** | Prompt-Init VLM FT (`RGB`) | $96.95\%$ | $0.9474$ | $0.9697$ |
| **Model 7: ResNet-18 (1.0s / Multi-View RGB)** | ImageNet Transfer | $97.27\%$ / $97.16\%$ | $0.9543$ / $0.9512$ | $0.9727$ / $0.9716$ |
| **Model 8: ConvNeXt-Tiny (Multi-View RGB)** | ImageNet Transfer (`RGB`) | $97.69\%$ | $0.9606$ | $0.9769$ |
| **Model 9: ViT-B/16 (Multi-View RGB)** | ImageNet Transfer (`RGB`) | $97.79\%$ | $0.9608$ | $0.9783$ |
| **Model 10: Bi-Modal Late Fusion (RF + ViT)** | Calibrated Late Fusion | $97.90\%$ | $0.9649$ | $0.9788$ |
| **Model 11: Early Fusion MLP (Frozen CLIP + 34 Physics)** | **0 GPU Fine-Tuning!** | $97.79\%$ | $0.9671$ | $0.9780$ |
| **Model 12: Tri-Modal Late Fusion (RF + ViT + CLIP)** | Calibrated Tri-Modal Fusion | $98.21\%$ | $0.9684$ | $0.9819$ |
| **CHAMPION: Hybrid Fusion (Early Physics-CLIP + ViT-B/16)** | **Early + Late Multimodal** | **$98.53\%$** (`14` errs) | **$0.9736$** | **$0.9852$** |

*Headline Scientific Discoveries:*
- **Multimodal Hybrid Fusion Breaks All Records (`0.9736` Macro-F1, `98.53%` Accuracy):** Combining `34` physical Omicron/spectrogram features + frozen CLIP embeddings + calibrated ImageNet `ViT-B/16` cut total test errors in half vs. RF (`28 → 14`) and by `33%` vs. `ViT-B/16` (`21 → 14`), pushing **every single one of the 14 classes above `0.909` F1** (`No_Glitch +0.0856`, `None_of_the_Above +0.0758`, `Extremely_Loud = 100%` recall).
- **Zero-GPU-Fine-Tuning Breakthrough:** Even with **100% frozen** CLIP weights, our 2-branch Multimodal MLP (`Frozen CLIP + 34 Physics`) achieved **`0.9671` Macro-F1**, outperforming full GPU-fine-tuned `ViT-B/16` (`0.9608`)!
- **Irreducible Bayes Label-Noise Floor (`0.53%`):** Only **`5 / 951` glitches** were missed by all 5 paradigms—and on **100% (`5/5`) of them**, RF, CNN, ViT, CLIP, and Hybrid Fusion unanimously predict the exact same alternative class.

---

## 3. Option B: Bidirectional Cross-Detector Transfer & Few-Shot Recovery (5-Seed Mean ± Std)

| Model (`H1 ↔ L1`, Seeds `42–46`) | `H1 → L1` (`0%`) | `H1 → L1` (`10%` / `20%`) | `L1 → H1` (`0%`) | `L1 → H1` (`10%` / `20%`) |
| :--- | :---: | :---: | :---: | :---: |
| **Feature Random Forest (34 feats)** | $0.6355 \pm 0.0147$ | — | $0.7101 \pm 0.0101$ | — |
| **Adapted Gravity Spy CNN (1.0s)** | $0.7357 \pm 0.0233$ | — | $0.7095 \pm 0.0347$ | — |
| **Step 16 Frozen CLIP Linear Probe (CPU)** | $0.6806 \pm 0.0289$ | $0.8082$ / $0.8148$ | $0.7063 \pm 0.0229$ | $0.7921$ / $0.7748$ |
| **Step 17 Multimodal Frozen CLIP + 34 Physics (CPU)** | $0.6908 \pm 0.0198$ | $0.8784$ / $0.8687$ | $0.7664 \pm 0.0166$ | $0.8541$ / **$0.8732$** |
| **Step 15 Fine-Tuned ResNet-18 (Multi-View RGB)** | **$0.8512 \pm 0.0322$** | **$0.9168$ / $0.9426$** | **$0.7805 \pm 0.0276$** | **$0.8609$ / $0.8742$** |

*Key Cross-Detector Takeaways:* Fine-tuning pretrained `ResNet-18 (Multi-View RGB)` closes **`91.7%`** of the `H1 → L1` domain gap using just `20%` (`351`) Livingston glitches (`0.8512 → 0.9426` vs. `0.9509` Native L1). Meanwhile, fusing **Frozen CLIP + 34 Physics Features on CPU** lifts `L1 → H1` zero-shot by `+0.060` (`0.7664`) and virtually ties GPU-fine-tuned `ResNet-18` at `20%` (`0.8732` vs. `0.8742`) with `3.6×` lower variance!

---

## 4. Repository Structure

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
│   └── experiments/ (run_baselines.py, run_tree_models.py, run_ablations.py, run_cnn.py,
│                     run_cross_detector.py, run_transfer_learning.py, run_vlm_clip.py, run_multimodal_fusion.py)
├── notebooks/
│   ├── 01_data_and_detector_eda.ipynb                  # Step 1: EDA & H1 vs L1 distribution analysis
│   ├── 02_feature_verification.ipynb                   # Step 4: Multi-view HDF5 & array verification
│   ├── 03_baseline_and_tree_models.ipynb               # Steps 7-9: Baselines, Decision Tree & Random Forest
│   ├── 04_cnn_training_colab.ipynb                     # Steps 11-12: Adapted Gravity Spy CNN vs. Features
│   ├── 05_cross_detector_evaluation.ipynb              # Step 13: Option B Cross-Detector Transfer (5 seeds)
│   ├── 06_ablations_and_failure_analysis.ipynb         # Steps 10 & 14: Feature Ablations & Visual Failure Analysis
│   ├── 07_transfer_learning_colab.ipynb                # Step 15: Pretrained Vision Transfer & Few-Shot Adaptation
│   ├── 08_vlm_clip_colab.ipynb                         # Step 16: Vision-Language Models (OpenAI CLIP ViT-B/16)
│   └── 09_multimodal_fusion_and_unified_analysis.ipynb # Step 17: Multimodal Physics-Vision Fusion & Unified Analysis
├── outputs/
│   ├── figures/     # 21 publication-grade 300-DPI PNG visualizations
│   ├── models/      # rf_best_model.joblib, clip_embeddings_*.npz, vit_b16_val_test_cache.npz
│   └── tables/      # All benchmark, ablation, cross-detector, transfer, VLM, fusion & failure CSVs
└── report/          # Final research paper & presentation assets
```
