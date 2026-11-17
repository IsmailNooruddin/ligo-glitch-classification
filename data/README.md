# Data Directory

Contains metadata, split registries, and cached feature matrices.

## Raw Files (kept in root or referenced here):
- `trainingsetv1d1.h5`: 3GB HDF5 archive containing 4 spectrogram duration views (0.5s, 1.0s, 2.0s, 4.0s) of shape (140, 170).
- `trainingset_v1d1_metadata.csv`: Tabular metadata (7,966 samples) containing `gravityspy_id`, labels, detector site (`ifo`), Omicron trigger attributes, and official splits.

## Processed Files (`data/processed/`):
- `shared_classes_metadata.csv`: Metadata subset filtered strictly to classes present in both H1 and L1.
- `split_indices.json`: Deterministic train/validation/test index mappings grouped by `gravityspy_id`.
- `engineered_features.csv`: Tabular cache of features extracted from spectrogram arrays + Omicron metadata.
