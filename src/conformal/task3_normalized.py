"""
Task 3: Normalized Conformal Classification

This module implements normalized conformal classification using a Mondrian approach
where categories are determined from difficulty estimations. This results in more
specific predictions for easier instances while maintaining coverage guarantees.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator


def estimate_difficulty(model, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
    """
    Estimate difficulty for each instance based on prediction confidence.
    
    For classification, we use the entropy of predicted probabilities as a difficulty measure.
    Higher entropy (more uncertainty) indicates higher difficulty.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classifier with predict_proba method
    X : pd.DataFrame
        Features
    y : pd.Series, optional
        True labels (not used, but kept for consistency)
    
    Returns:
    --------
    difficulty_scores : np.ndarray
        Array of difficulty scores (higher = more difficult)
    """
    probs = model.predict_proba(X)
    
    # Use entropy as difficulty measure
    # Entropy = -sum(p * log(p)) for each class probability
    # Higher entropy = more uncertainty = higher difficulty
    epsilon = 1e-10  # Avoid log(0)
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
    
    return entropy


def discretize_difficulty(difficulty_scores: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Discretize difficulty scores into bins for Mondrian categories.
    
    Parameters:
    -----------
    difficulty_scores : np.ndarray
        Array of difficulty scores
    n_bins : int
        Number of difficulty bins
    
    Returns:
    --------
    difficulty_bins : np.ndarray
        Array of bin indices (0 to n_bins-1)
    """
    # Use quantile-based binning to ensure balanced bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(difficulty_scores, percentiles)
    bin_edges[0] = -np.inf  # Ensure all values are included
    bin_edges[-1] = np.inf
    
    difficulty_bins = np.digitize(difficulty_scores, bin_edges) - 1
    difficulty_bins = np.clip(difficulty_bins, 0, n_bins - 1)
    
    return difficulty_bins


def normalized_conformal_classification(
    model,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    X_test: pd.DataFrame,
    significance_level: float,
    n_difficulty_bins: int = 5
) -> Tuple[List[List[int]], Dict]:
    """
    Implement normalized conformal classification using difficulty-based Mondrian categories.
    
    This method groups instances by estimated difficulty and computes category-specific
    thresholds, resulting in more specific predictions for easier instances.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classifier
    X_cal : pd.DataFrame
        Calibration features
    y_cal : pd.Series
        Calibration labels
    X_test : pd.DataFrame
        Test features
    significance_level : float
        Significance level ε
    n_difficulty_bins : int
        Number of difficulty bins for Mondrian categories
    
    Returns:
    --------
    prediction_sets : List[List[int]]
        Prediction sets for each test example
    category_thresholds : Dict
        Dictionary mapping (difficulty_bin, class) -> threshold
    """
    # Estimate difficulty for calibration and test sets
    cal_difficulty = estimate_difficulty(model, X_cal)
    test_difficulty = estimate_difficulty(model, X_test)
    
    # Discretize difficulty into bins
    cal_difficulty_bins = discretize_difficulty(cal_difficulty, n_difficulty_bins)
    test_difficulty_bins = discretize_difficulty(test_difficulty, n_difficulty_bins)
    
    # Get predicted probabilities
    cal_probs = model.predict_proba(X_cal)
    test_probs = model.predict_proba(X_test)
    
    n_classes = cal_probs.shape[1]
    
    # Compute category-specific thresholds
    # Category = (difficulty_bin, class)
    category_thresholds = {}
    
    for diff_bin in range(n_difficulty_bins):
        for c in range(n_classes):
            # Get calibration examples in this difficulty bin with this class
            bin_mask = (cal_difficulty_bins == diff_bin)
            class_mask = np.array([
                int(y_cal.iloc[i]) == c if hasattr(y_cal, 'iloc') else int(y_cal[i]) == c
                for i in range(len(y_cal))
            ])
            category_mask = bin_mask & class_mask
            category_indices = np.where(category_mask)[0]
            
            if len(category_indices) == 0:
                # If no examples in this category, use a very high threshold
                category_thresholds[(diff_bin, c)] = 1.0
                continue
            
            # Compute non-conformity scores for this category: 1 - P(c | x)
            category_scores = []
            for idx in category_indices:
                score = 1 - cal_probs[idx, c]
                category_scores.append(score)
            
            category_scores = np.array(category_scores)
            
            # Compute category-specific threshold with finite-sample correction
            n_category = len(category_scores)
            sorted_scores = np.sort(category_scores)
            quantile_idx = int(np.ceil((n_category + 1) * (1 - significance_level))) - 1
            quantile_idx = min(quantile_idx, n_category - 1)
            category_thresholds[(diff_bin, c)] = sorted_scores[quantile_idx]
    
    # Form prediction sets using category-specific thresholds
    n_test = len(X_test)
    prediction_sets = []
    
    for i in range(n_test):
        prediction_set = []
        diff_bin = test_difficulty_bins[i]
        
        for c in range(n_classes):
            # Include class c if its non-conformity score <= category-specific threshold
            nonconf_score = 1 - test_probs[i, c]
            threshold = category_thresholds.get((diff_bin, c), 1.0)
            if nonconf_score <= threshold:
                prediction_set.append(c)
        
        prediction_sets.append(prediction_set)
    
    return prediction_sets, category_thresholds


def evaluate_normalized_predictions(
    prediction_sets: List[List[int]],
    y_true: pd.Series,
    difficulty_scores: np.ndarray = None
) -> Dict:
    """
    Evaluate normalized conformal predictions, optionally broken down by difficulty.
    
    Parameters:
    -----------
    prediction_sets : List[List[int]]
        Prediction sets for each test example
    y_true : pd.Series
        True labels
    difficulty_scores : np.ndarray, optional
        Difficulty scores for each test example
    
    Returns:
    --------
    Dictionary with evaluation metrics, optionally including difficulty-specific metrics
    """
    n = len(prediction_sets)
    errors = 0
    total_size = 0
    
    for i, pred_set in enumerate(prediction_sets):
        true_class = int(y_true.iloc[i]) if hasattr(y_true, 'iloc') else int(y_true[i])
        
        if true_class not in pred_set:
            errors += 1
        
        total_size += len(pred_set)
    
    error_rate = errors / n
    avg_set_size = total_size / n
    
    result = {
        'error_rate': error_rate,
        'avg_set_size': avg_set_size
    }
    
    # If difficulty scores provided, compute difficulty-specific metrics
    if difficulty_scores is not None:
        # Split into low/medium/high difficulty
        low_threshold = np.percentile(difficulty_scores, 33)
        high_threshold = np.percentile(difficulty_scores, 67)
        
        low_mask = difficulty_scores <= low_threshold
        medium_mask = (difficulty_scores > low_threshold) & (difficulty_scores <= high_threshold)
        high_mask = difficulty_scores > high_threshold
        
        for mask, name in [(low_mask, 'low'), (medium_mask, 'medium'), (high_mask, 'high')]:
            if np.sum(mask) > 0:
                mask_errors = sum(
                    1 for i in np.where(mask)[0]
                    if int(y_true.iloc[i]) not in prediction_sets[i]
                )
                mask_avg_size = np.mean([len(prediction_sets[i]) for i in np.where(mask)[0]])
                result[f'{name}_difficulty_error_rate'] = mask_errors / np.sum(mask)
                result[f'{name}_difficulty_avg_set_size'] = mask_avg_size
    
    return result

