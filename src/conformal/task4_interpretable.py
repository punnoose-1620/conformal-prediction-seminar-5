"""
Task 4: Interpretable Conformal Classifiers

This module implements conformal classification trees, where each leaf returns
a conformal prediction set. This combines uncertainty quantification with
interpretability through decision tree structure.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin


class ConformalClassificationTree(BaseEstimator, ClassifierMixin):
    """
    A decision tree where each leaf returns a conformal prediction set.
    
    The tree structure provides interpretability, while conformal prediction
    provides coverage guarantees. Can use either global or Mondrian (leaf-specific)
    calibration.
    """
    
    def __init__(
        self,
        max_depth: int = 5,
        mondrian: bool = True,
        significance_level: float = 0.05,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        max_depth : int
            Maximum depth of the tree (shallower = more interpretable)
        mondrian : bool
            If True, use Mondrian calibration (leaf-specific thresholds)
            If False, use global calibration
        significance_level : float
            Significance level ε for conformal prediction
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required in a leaf
        random_state : int
            Random seed
        """
        self.max_depth = max_depth
        self.mondrian = mondrian
        self.significance_level = significance_level
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.tree_ = None
        self.leaf_thresholds_ = {}  # For Mondrian: leaf_id -> threshold
        self.global_threshold_ = None  # For global calibration
        self.n_classes_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the conformal classification tree.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training labels
        """
        # Split into proper training and calibration
        n = len(X)
        n_cal = n // 2
        cal_idx = np.random.choice(n, size=n_cal, replace=False)
        proper_train_idx = np.setdiff1d(np.arange(n), cal_idx)
        
        X_proper_train = X.iloc[proper_train_idx]
        X_cal = X.iloc[cal_idx]
        y_proper_train = y.iloc[proper_train_idx]
        y_cal = y.iloc[cal_idx]
        
        # Train the decision tree
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_.fit(X_proper_train, y_proper_train)
        
        self.n_classes_ = len(np.unique(y))
        
        # Calibrate using calibration set
        if self.mondrian:
            self._calibrate_mondrian(X_cal, y_cal)
        else:
            self._calibrate_global(X_cal, y_cal)
        
        return self
    
    def _calibrate_global(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Calibrate using global threshold across all leaves."""
        # Get leaf assignments for calibration examples
        cal_leaves = self.tree_.apply(X_cal.values)
        cal_probs = self.tree_.predict_proba(X_cal.values)
        
        # Compute non-conformity scores: 1 - P(y_true | x)
        n_cal = len(X_cal)
        cal_scores = []
        for i in range(n_cal):
            true_class = int(y_cal.iloc[i]) if hasattr(y_cal, 'iloc') else int(y_cal[i])
            score = 1 - cal_probs[i, true_class]
            cal_scores.append(score)
        
        cal_scores = np.array(cal_scores)
        
        # Compute global threshold
        n = len(cal_scores)
        sorted_scores = np.sort(cal_scores)
        quantile_idx = int(np.ceil((n + 1) * (1 - self.significance_level))) - 1
        quantile_idx = min(quantile_idx, n - 1)
        self.global_threshold_ = sorted_scores[quantile_idx]
    
    def _calibrate_mondrian(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """Calibrate using Mondrian approach with leaf-specific thresholds."""
        # Get leaf assignments for calibration examples
        cal_leaves = self.tree_.apply(X_cal.values)
        cal_probs = self.tree_.predict_proba(X_cal.values)
        
        unique_leaves = np.unique(cal_leaves)
        self.leaf_thresholds_ = {}
        
        for leaf_id in unique_leaves:
            # Get calibration examples in this leaf
            leaf_mask = (cal_leaves == leaf_id)
            leaf_indices = np.where(leaf_mask)[0]
            
            if len(leaf_indices) < self.min_samples_leaf:
                # If too few examples, use a high threshold
                self.leaf_thresholds_[leaf_id] = 1.0
                continue
            
            # Compute non-conformity scores for this leaf: 1 - P(y_true | x)
            leaf_scores = []
            for idx in leaf_indices:
                true_class = int(y_cal.iloc[idx]) if hasattr(y_cal, 'iloc') else int(y_cal[idx])
                score = 1 - cal_probs[idx, true_class]
                leaf_scores.append(score)
            
            leaf_scores = np.array(leaf_scores)
            
            # Compute leaf-specific threshold
            n_leaf = len(leaf_scores)
            sorted_scores = np.sort(leaf_scores)
            quantile_idx = int(np.ceil((n_leaf + 1) * (1 - self.significance_level))) - 1
            quantile_idx = min(quantile_idx, n_leaf - 1)
            self.leaf_thresholds_[leaf_id] = sorted_scores[quantile_idx]
    
    def predict_conformal(self, X: pd.DataFrame) -> List[List[int]]:
        """
        Predict conformal prediction sets.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Test features
        
        Returns:
        --------
        prediction_sets : List[List[int]]
            Prediction sets for each test example
        """
        # Get leaf assignments and probabilities
        test_leaves = self.tree_.apply(X.values)
        test_probs = self.tree_.predict_proba(X.values)
        
        n_test = len(X)
        prediction_sets = []
        
        for i in range(n_test):
            leaf_id = test_leaves[i]
            
            # Get threshold for this leaf
            if self.mondrian:
                threshold = self.leaf_thresholds_.get(leaf_id, 1.0)
            else:
                threshold = self.global_threshold_
            
            # Form prediction set: include class c if P(c|x) >= 1 - threshold
            prediction_set = []
            for c in range(self.n_classes_):
                if test_probs[i, c] >= 1 - threshold:
                    prediction_set.append(c)
            
            prediction_sets.append(prediction_set)
        
        return prediction_sets
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Standard point prediction (most likely class)."""
        return self.tree_.predict(X.values)
    
    def visualize_tree(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (20, 10)
    ):
        """
        Visualize the decision tree structure.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features. Must match the number of features the tree was trained on.
            If None or length mismatch, generic names will be used.
        class_names : list, optional
            Names of classes
        figsize : tuple
            Figure size
        """
        if self.tree_ is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        
        # Get the actual number of features the tree was trained on
        n_features = self.tree_.n_features_in_
        
        # Validate feature_names length
        if feature_names is not None and len(feature_names) != n_features:
            print(f"Warning: feature_names length ({len(feature_names)}) doesn't match "
                  f"number of features ({n_features}). Using generic feature names.")
            feature_names = None
        
        # Use generic names if not provided or invalid
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_tree(
            self.tree_,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            ax=ax,
            fontsize=10
        )
        
        plt.title(f"Conformal Classification Tree (Mondrian={self.mondrian}, ε={self.significance_level})")
        plt.tight_layout()
        plt.show()
    
    def get_leaf_info(self) -> pd.DataFrame:
        """
        Get information about each leaf, including thresholds and statistics.
        
        Returns:
        --------
        DataFrame with leaf information
        """
        if self.tree_ is None:
            raise ValueError("Tree not fitted yet. Call fit() first.")
        
        tree = self.tree_.tree_
        leaf_info = []
        
        def traverse(node_id, depth=0):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                n_samples = tree.n_node_samples[node_id]
                value = tree.value[node_id][0]
                threshold = self.leaf_thresholds_.get(node_id, self.global_threshold_) if self.mondrian else self.global_threshold_
                
                leaf_info.append({
                    'leaf_id': node_id,
                    'depth': depth,
                    'n_samples': n_samples,
                    'threshold': threshold,
                    'class_distribution': value / n_samples if n_samples > 0 else value
                })
            else:
                # Internal node
                traverse(tree.children_left[node_id], depth + 1)
                traverse(tree.children_right[node_id], depth + 1)
        
        traverse(0)
        return pd.DataFrame(leaf_info)

