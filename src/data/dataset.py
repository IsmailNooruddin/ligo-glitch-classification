"""
Module: dataset.py
Purpose: HDF5 Spectrogram Loader and PyTorch Dataset for LIGO Glitch Data.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Spectrogram matrices are stored inside a 3 GB HDF5 file (`trainingsetv1d1.h5`).
This module provides clean, efficient functions to read the 2D arrays, retrieve all 4 duration views
(0.5s, 1.0s, 2.0s, 4.0s), and wrap them in a standard PyTorch `Dataset` ready for deep learning.
"""

import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Standard duration keys mapping
DURATION_KEYS = {
    "0.5s": "0.5.png",
    "1.0s": "1.0.png",
    "2.0s": "2.0.png",
    "4.0s": "4.0.png"
}


def get_class_mapping(classes_list):
    """
    Creates bidirectional mapping dictionaries between class name strings and integer indices.

    Parameters:
    -----------
    classes_list : list of str
        List of unique class names.

    Returns:
    --------
    class_to_idx : dict
        Maps class string (e.g. 'Blip') to integer (0..N-1).
    idx_to_class : dict
        Maps integer (0..N-1) back to class string.
    """
    sorted_classes = sorted(classes_list)
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
    idx_to_class = {idx: name for idx, name in enumerate(sorted_classes)}
    return class_to_idx, idx_to_class


def load_spectrogram(h5_file, label, sample_type, gravityspy_id, view="1.0s"):
    """
    Retrieves a single 2D spectrogram array of shape (140, 170) from an open HDF5 file.

    Parameters:
    -----------
    h5_file : h5py.File
        An active open h5py File object.
    label : str
        The true glitch class name (e.g. 'Blip').
    sample_type : str
        The original partition in the H5 hierarchy ('train', 'validation', 'test').
    gravityspy_id : str
        The 10-character unique glitch identifier.
    view : str, default="1.0s"
        The desired duration view ('0.5s', '1.0s', '2.0s', or '4.0s').

    Returns:
    --------
    array : np.ndarray
        2D float32 array of shape (140, 170).
    """
    dataset_key = DURATION_KEYS.get(view, "1.0.png")
    h5_path = f"{label}/{sample_type}/{gravityspy_id}/{dataset_key}"

    # Read data from HDF5
    raw_array = h5_file[h5_path][:]

    # The raw shape in HDF5 is (1, 140, 170); squeeze down to (140, 170)
    if raw_array.ndim == 3 and raw_array.shape[0] == 1:
        raw_array = raw_array.squeeze(axis=0)

    return raw_array.astype(np.float32)


def load_all_views(h5_file, label, sample_type, gravityspy_id):
    """
    Retrieves all 4 duration views for a single glitch event.

    Returns:
    --------
    views_dict : dict
        Dictionary mapping view names ('0.5s', '1.0s', '2.0s', '4.0s') to 2D arrays.
    """
    views_dict = {}
    for view_name in ["0.5s", "1.0s", "2.0s", "4.0s"]:
        views_dict[view_name] = load_spectrogram(h5_file, label, sample_type, gravityspy_id, view=view_name)
    return views_dict


class LIGOGlitchDataset(Dataset):
    """
    PyTorch Dataset for LIGO Glitch Spectrograms.
    Supports single-view (e.g. 1.0s) or multi-channel 4-view loading.
    
    Uses lazy HDF5 file initialization so that PyTorch DataLoader multi-processing
    workers each maintain an independent, thread-safe file handle.
    """
    def __init__(self, h5_path, metadata_df, view="1.0s", class_to_idx=None, transform=None):
        """
        Parameters:
        -----------
        h5_path : str
            Filepath to 'trainingsetv1d1.h5'.
        metadata_df : pd.DataFrame
            DataFrame containing samples to load (must have 'gravityspy_id', 'label', 'sample_type').
        view : str, default="1.0s"
            Which view to load ('0.5s', '1.0s', '2.0s', '4.0s', or 'all' for 4 channels).
        class_to_idx : dict, optional
            Mapping from class name to integer index. If None, generated automatically.
        transform : callable, optional
            Optional PyTorch transform applied to image tensors.
        """
        self.h5_path = os.path.abspath(h5_path)
        self.metadata = metadata_df.reset_index(drop=True)
        self.view = view
        self.transform = transform

        # Build or assign class-to-index mapping
        if class_to_idx is None:
            unique_classes = sorted(self.metadata['label'].unique().tolist())
            self.class_to_idx, self.idx_to_class = get_class_mapping(unique_classes)
        else:
            self.class_to_idx = class_to_idx
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # HDF5 file handle is set lazily per process/thread
        self.h5_file = None

    def _get_h5_file(self):
        """Lazily opens HDF5 file handle for the current process."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        return self.h5_file

    def __len__(self):
        """Total number of samples in this dataset partition."""
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample tensor and its label index.

        Returns:
        --------
        image_tensor : torch.Tensor
            If view is single: shape (1, 140, 170).
            If view is 'all': shape (4, 140, 170).
        label_idx : int
            Integer class label index.
        gravityspy_id : str
            The glitch identifier.
        """
        row = self.metadata.iloc[idx]
        label_str = row['label']
        sample_type = row['sample_type']
        glitch_id = row['gravityspy_id']

        h5_f = self._get_h5_file()

        if self.view == "all":
            # Stack all 4 duration views as 4 channels: shape (4, 140, 170)
            channels = []
            for v_name in ["0.5s", "1.0s", "2.0s", "4.0s"]:
                arr = load_spectrogram(h5_f, label_str, sample_type, glitch_id, view=v_name)
                channels.append(arr)
            img_np = np.stack(channels, axis=0)  # Shape: (4, 140, 170)
        else:
            # Single view: shape (1, 140, 170)
            arr = load_spectrogram(h5_f, label_str, sample_type, glitch_id, view=self.view)
            img_np = np.expand_dims(arr, axis=0)  # Shape: (1, 140, 170)

        # Convert to PyTorch float tensor
        img_tensor = torch.from_numpy(img_np).float()

        # Apply optional transforms
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        # Look up integer label
        label_idx = self.class_to_idx[label_str]

        return img_tensor, label_idx, glitch_id

    def __del__(self):
        """Closes HDF5 file handle upon garbage collection."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
