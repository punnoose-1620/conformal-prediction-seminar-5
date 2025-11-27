"""
Task 2: Class-Conditional Conformal Prediction Evaluation

This module provides functions to evaluate and compare standard conformal prediction
with class-conditional conformal prediction (CCCP), reporting class-specific error rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import StratifiedKFold


def evaluate_class_conditional_comparison(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    significance_level: float,
    standard_conformal_fn,
    class_conditional_conformal_fn,
    n_cal_split: float = 0.5
) -> Dict[str, Any]:
    """
    Compare standard and class-conditional conformal prediction on a single train/test split.
    
    Parameters:
    -----------
    model : sklearn classifier
        The base classifier to use
    X_train, y_train : Training data
    X_test, y_test : Test data
    significance_level : float
        Significance level ε
    standard_conformal_fn : callable
        Function for standard conformal prediction
    class_conditional_conformal_fn : callable
        Function for class-conditional conformal prediction
    n_cal_split : float
        Fraction of training data to use for calibration (default 0.5)
    
    Returns:
    --------
    Dictionary with comparison results including:
    - standard_error_rate: Overall error rate for standard CP
    - standard_class_errors: Class-specific error rates for standard CP
    - cccp_error_rate: Overall error rate for CCCP
    - cccp_class_errors: Class-specific error rates for CCCP
    - standard_avg_set_size: Average set size for standard CP
    - cccp_avg_set_size: Average set size for CCCP
    """
    # Split training into proper training and calibration
    n_train = len(X_train)
    n_cal = int(n_train * n_cal_split)
    cal_idx = np.random.choice(n_train, size=n_cal, replace=False)
    proper_train_idx = np.setdiff1d(np.arange(n_train), cal_idx)
    
    X_proper_train = X_train.iloc[proper_train_idx]
    X_cal = X_train.iloc[cal_idx]
    y_proper_train = y_train.iloc[proper_train_idx]
    y_cal = y_train.iloc[cal_idx]
    
    # Train model
    model.fit(X_proper_train, y_proper_train)
    
    # Standard conformal prediction
    standard_pred_sets, _ = standard_conformal_fn(
        model, X_cal, y_cal, X_test, significance_level
    )
    
    # Class-conditional conformal prediction
    cccp_pred_sets, _ = class_conditional_conformal_fn(
        model, X_cal, y_cal, X_test, significance_level
    )
    
    # Evaluate standard CP
    standard_results = _evaluate_with_class_errors(standard_pred_sets, y_test)
    
    # Evaluate CCCP
    cccp_results = _evaluate_with_class_errors(cccp_pred_sets, y_test)
    
    return {
        'standard_error_rate': standard_results['error_rate'],
        'standard_class_errors': standard_results['class_error_rates'],
        'standard_avg_set_size': standard_results['avg_set_size'],
        'cccp_error_rate': cccp_results['error_rate'],
        'cccp_class_errors': cccp_results['class_error_rates'],
        'cccp_avg_set_size': cccp_results['avg_set_size'],
    }


def _evaluate_with_class_errors(prediction_sets: List[List[int]], y_true: pd.Series) -> Dict:
    """Helper function to evaluate predictions with class-specific error rates."""
    n = len(prediction_sets)
    errors = 0
    total_size = 0
    
    class_errors = {}
    class_counts = {}
    
    for i, pred_set in enumerate(prediction_sets):
        true_class = int(y_true.iloc[i]) if hasattr(y_true, 'iloc') else int(y_true[i])
        
        if true_class not in class_counts:
            class_counts[true_class] = 0
            class_errors[true_class] = 0
        
        class_counts[true_class] += 1
        
        if true_class not in pred_set:
            errors += 1
            class_errors[true_class] += 1
        
        total_size += len(pred_set)
    
    error_rate = errors / n
    avg_set_size = total_size / n
    
    class_error_rates = {}
    for c in class_counts:
        class_error_rates[c] = class_errors[c] / class_counts[c] if class_counts[c] > 0 else 0.0
    
    return {
        'error_rate': error_rate,
        'class_error_rates': class_error_rates,
        'avg_set_size': avg_set_size
    }


def run_class_conditional_evaluation(
    models: Dict[str, Any],
    datasets: List[Tuple[str, pd.DataFrame, pd.Series]],
    significance_levels: List[float],
    standard_conformal_fn,
    class_conditional_conformal_fn,
    n_folds: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run full evaluation comparing standard and class-conditional conformal prediction.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model_name -> model_instance
    datasets : list
        List of (dataset_name, X, y) tuples
    significance_levels : list
        List of significance levels to test
    standard_conformal_fn : callable
        Function for standard conformal prediction
    class_conditional_conformal_fn : callable
        Function for class-conditional conformal prediction
    n_folds : int
        Number of CV folds
    random_state : int
        Random seed
    
    Returns:
    --------
    DataFrame with results for each dataset, model, fold, and significance level
    """
    results = []
    
    for dataset_name, X, y in datasets:
        # Skip if not binary classification
        if len(y.unique()) != 2:
            continue
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            for model_name, model_class in models.items():
                # Create fresh model instance
                model = type(model_class)(**model_class.get_params())
                
                for sig_level in significance_levels:
                    comparison = evaluate_class_conditional_comparison(
                        model, X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                        sig_level, standard_conformal_fn, class_conditional_conformal_fn
                    )
                    
                    # Store results
                    for class_label in comparison['standard_class_errors'].keys():
                        results.append({
                            'dataset': dataset_name,
                            'fold': fold_idx,
                            'model': model_name,
                            'significance_level': sig_level,
                            'class': class_label,
                            'standard_error_rate': comparison['standard_error_rate'],
                            'standard_class_error_rate': comparison['standard_class_errors'][class_label],
                            'cccp_error_rate': comparison['cccp_error_rate'],
                            'cccp_class_error_rate': comparison['cccp_class_errors'][class_label],
                            'standard_avg_set_size': comparison['standard_avg_set_size'],
                            'cccp_avg_set_size': comparison['cccp_avg_set_size'],
                        })
    
    return pd.DataFrame(results)

