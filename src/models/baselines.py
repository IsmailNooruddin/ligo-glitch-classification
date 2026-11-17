"""
Module: baselines.py
Purpose: Lower-bound baseline models for LIGO glitch classification.
Author: L4 Group 23 (Ismail Nooruddin, Munir Tariq, Ramsha Arshad)

Why this module is essential:
Establishes the minimal performance floor as described in our Project Proposal:
1. Majority Class Guess: Simply predicts the most common class in the training data (e.g. 'Blip').
   This demonstrates why high accuracy can be an illusion when severe class imbalance is present.
2. Raw-Pixel Decision Tree: Flattens the 140x170 spectrogram into a 23,800-dimensional 1D vector
   and trains an unregularized Decision Tree, demonstrating the failure of raw pixels without feature engineering.
"""

from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MajorityClassBaseline:
    """
    A non-learning dummy baseline classifier that always predicts the most
    frequent class observed in the training dataset.
    """
    def __init__(self):
        self.majority_class_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Learns the most frequent label in y.

        Parameters:
        -----------
        X : array-like
            Training features (ignored by this baseline).
        y : array-like
            Ground-truth training labels.
        """
        y_list = list(y)
        counts = Counter(y_list)
        # Find the label with highest count
        self.majority_class_ = counts.most_common(1)[0][0]
        self.classes_ = sorted(list(counts.keys()))
        return self

    def predict(self, X):
        """
        Returns the majority class for all samples in X.

        Parameters:
        -----------
        X : array-like
            Samples to predict.

        Returns:
        --------
        predictions : list
            List of majority class strings with length equal to len(X).
        """
        n_samples = len(X)
        return [self.majority_class_] * n_samples

    def predict_proba(self, X):
        """
        Returns 1.0 probability for the majority class, 0.0 for all others.
        """
        n_samples = len(X)
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=np.float32)
        maj_idx = self.classes_.index(self.majority_class_)
        proba[:, maj_idx] = 1.0
        return proba


class RawPixelDecisionTree:
    """
    Decision tree trained directly on flattened 23,800-dimensional raw pixel vectors.
    """
    def __init__(self, max_depth=10, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def fit(self, X_pixels, y):
        """
        Fits the decision tree on flattened pixel arrays.
        """
        return self.model.fit(X_pixels, y)

    def predict(self, X_pixels):
        """
        Predicts glitch classes from raw pixels.
        """
        return self.model.predict(X_pixels)
