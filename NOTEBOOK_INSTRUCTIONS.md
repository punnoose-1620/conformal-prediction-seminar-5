# Instructions for Adding Tasks 2-4 to Conformal_Classification.ipynb

This document provides instructions for adding cells to your notebook to complete Tasks 2, 3, and 4.

## Task 2: Class-Conditional Conformal Prediction Evaluation

### Add these cells after Cell 14:

**Cell 15 (Markdown):**
```markdown
## Task 2: Evaluation - Standard vs Class-Conditional

Now we'll compare standard conformal prediction with class-conditional conformal prediction,
reporting class-specific error rates to see how CCCP addresses imbalanced error rates.
```

**Cell 16 (Code):**
```python
# Import Task 2 evaluation functions
from src.conformal.task2_evaluation import (
    evaluate_class_conditional_comparison,
    run_class_conditional_evaluation
)

# Import the conformal functions from earlier cells
# (These should already be defined in your notebook)
```

**Cell 17 (Code):**
```python
# Run comparison evaluation on selected datasets
task2_results = []

selected_datasets_task2 = dataset_files[:10]  # Use 10 datasets

for dataset_name in tqdm(selected_datasets_task2, desc="Task 2: Datasets"):
    dataset_path = twoclass_dir / dataset_name
    
    try:
        X, y = load_dataset(dataset_path)
        
        if len(X) < 20 or len(y.unique()) != 2:
            continue
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            for model_name, model_class in models.items():
                model = type(model_class)(**model_class.get_params())
                
                for sig_level in significance_levels:
                    comparison = evaluate_class_conditional_comparison(
                        model, X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                        sig_level, split_conformal_classification, class_conditional_conformal_classification
                    )
                    
                    # Store results for each class
                    for class_label in comparison['standard_class_errors'].keys():
                        task2_results.append({
                            'dataset': dataset_name,
                            'fold': fold_idx,
                            'model': model_name,
                            'significance_level': sig_level,
                            'class': class_label,
                            'standard_error_rate': comparison['standard_error_rate'],
                            'standard_class_error': comparison['standard_class_errors'][class_label],
                            'cccp_error_rate': comparison['cccp_error_rate'],
                            'cccp_class_error': comparison['cccp_class_errors'][class_label],
                            'standard_avg_set_size': comparison['standard_avg_set_size'],
                            'cccp_avg_set_size': comparison['cccp_avg_set_size'],
                        })
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

task2_df = pd.DataFrame(task2_results)
print(f"Task 2 results collected: {len(task2_df)} rows")
```

**Cell 18 (Code):**
```python
# Analyze class-specific error rates
print("=" * 80)
print("TASK 2: CLASS-SPECIFIC ERROR RATES COMPARISON")
print("=" * 80)

# Compare standard vs CCCP class-specific error rates
for sig_level in significance_levels:
    print(f"\nSignificance Level: ε={sig_level} (Confidence: {1-sig_level:.0%})")
    sig_data = task2_df[task2_df['significance_level'] == sig_level]
    
    for model_name in models.keys():
        model_data = sig_data[sig_data['model'] == model_name]
        
        print(f"\n{model_name}:")
        print(f"  Standard CP - Class 0 error: {model_data['standard_class_error'].mean():.4f}")
        print(f"  Standard CP - Class 1 error: {model_data[model_data['class']==1]['standard_class_error'].mean():.4f}")
        print(f"  CCCP - Class 0 error: {model_data[model_data['class']==0]['cccp_class_error'].mean():.4f}")
        print(f"  CCCP - Class 1 error: {model_data[model_data['class']==1]['cccp_class_error'].mean():.4f}")
        
        # Compute error rate imbalance
        std_errors = model_data.groupby('class')['standard_class_error'].mean()
        cccp_errors = model_data.groupby('class')['cccp_class_error'].mean()
        
        std_imbalance = abs(std_errors[0] - std_errors[1])
        cccp_imbalance = abs(cccp_errors[0] - cccp_errors[1])
        
        print(f"  Standard CP error imbalance: {std_imbalance:.4f}")
        print(f"  CCCP error imbalance: {cccp_imbalance:.4f}")
```

**Cell 19 (Code):**
```python
# Visualization: Class-specific error rates
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sig_level in enumerate(significance_levels):
    ax = axes[idx]
    sig_data = task2_df[task2_df['significance_level'] == sig_level]
    
    # Group by class and method
    classes = sorted(sig_data['class'].unique())
    x_pos = np.arange(len(classes))
    width = 0.35
    
    standard_errors = [sig_data[(sig_data['class']==c)]['standard_class_error'].mean() for c in classes]
    cccp_errors = [sig_data[(sig_data['class']==c)]['cccp_class_error'].mean() for c in classes]
    
    ax.bar(x_pos - width/2, standard_errors, width, label='Standard CP', alpha=0.7)
    ax.bar(x_pos + width/2, cccp_errors, width, label='CCCP', alpha=0.7)
    ax.axhline(sig_level, color='r', linestyle='--', label=f'Target (ε={sig_level})')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Error Rate')
    ax.set_title(f'Class-Specific Error Rates (ε={sig_level})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Class {c}' for c in classes])
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Cell 20 (Markdown):**
```markdown
### Task 2 Commentary

Class-conditional conformal prediction (CCCP) addresses the issue where standard conformal prediction
can lead to imbalanced error rates across classes. By providing separate coverage guarantees for each
class using a Mondrian approach, CCCP ensures that each class gets approximately the target error rate.

Key observations:
- Standard CP may have very different error rates for different classes
- CCCP provides more balanced error rates across classes
- This is particularly important when classes are imbalanced or have different costs
```

---

## Task 3: Normalized Conformal Classification

### Add these cells after Task 2:

**Cell 21 (Markdown):**
```markdown
## Task 3: Normalized Conformal Classification

Normalized conformal classification uses difficulty estimation to create Mondrian categories,
resulting in more specific predictions for easier instances while maintaining coverage guarantees.
```

**Cell 22 (Code):**
```python
# Import Task 3 functions
from src.conformal.task3_normalized import (
    normalized_conformal_classification,
    evaluate_normalized_predictions,
    estimate_difficulty
)
```

**Cell 23 (Code):**
```python
# Run normalized conformal classification evaluation
task3_results = []

selected_datasets_task3 = dataset_files[:10]

for dataset_name in tqdm(selected_datasets_task3, desc="Task 3: Datasets"):
    dataset_path = twoclass_dir / dataset_name
    
    try:
        X, y = load_dataset(dataset_path)
        
        if len(X) < 20 or len(y.unique()) != 2:
            continue
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Split training into proper training and calibration
            n_train = len(X_train_fold)
            n_cal = n_train // 2
            cal_idx = np.random.choice(n_train, size=n_cal, replace=False)
            proper_train_idx = np.setdiff1d(np.arange(n_train), cal_idx)
            
            X_proper_train = X_train_fold.iloc[proper_train_idx]
            X_cal = X_train_fold.iloc[cal_idx]
            y_proper_train = y_train_fold.iloc[proper_train_idx]
            y_cal = y_train_fold.iloc[cal_idx]
            
            for model_name, model_class in models.items():
                model = type(model_class)(**model_class.get_params())
                model.fit(X_proper_train, y_proper_train)
                
                # Estimate difficulty for test set
                test_difficulty = estimate_difficulty(model, X_test_fold)
                
                for sig_level in significance_levels:
                    # Normalized conformal prediction
                    norm_pred_sets, _ = normalized_conformal_classification(
                        model, X_cal, y_cal, X_test_fold, sig_level, n_difficulty_bins=5
                    )
                    
                    # Standard conformal prediction for comparison
                    std_pred_sets, _ = split_conformal_classification(
                        model, X_cal, y_cal, X_test_fold, sig_level
                    )
                    
                    # Evaluate normalized
                    norm_eval = evaluate_normalized_predictions(
                        norm_pred_sets, y_test_fold, test_difficulty
                    )
                    
                    # Evaluate standard
                    std_eval = evaluate_conformal_predictions(std_pred_sets, y_test_fold)
                    
                    task3_results.append({
                        'dataset': dataset_name,
                        'fold': fold_idx,
                        'model': model_name,
                        'significance_level': sig_level,
                        'normalized_error_rate': norm_eval['error_rate'],
                        'normalized_avg_set_size': norm_eval['avg_set_size'],
                        'standard_error_rate': std_eval['error_rate'],
                        'standard_avg_set_size': std_eval['avg_set_size'],
                        'efficiency_gain': std_eval['avg_set_size'] - norm_eval['avg_set_size'],
                    })
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

task3_df = pd.DataFrame(task3_results)
print(f"Task 3 results collected: {len(task3_df)} rows")
```

**Cell 24 (Code):**
```python
# Analyze normalized conformal classification results
print("=" * 80)
print("TASK 3: NORMALIZED CONFORMAL CLASSIFICATION RESULTS")
print("=" * 80)

# Compare efficiency (average set size)
summary_task3 = task3_df.groupby(['model', 'significance_level']).agg({
    'normalized_avg_set_size': 'mean',
    'standard_avg_set_size': 'mean',
    'efficiency_gain': 'mean',
    'normalized_error_rate': 'mean',
    'standard_error_rate': 'mean'
}).round(4)

print("\nAverage Set Size Comparison:")
print(summary_task3[['standard_avg_set_size', 'normalized_avg_set_size', 'efficiency_gain']])
print("\nError Rates (should be similar):")
print(summary_task3[['standard_error_rate', 'normalized_error_rate']])
```

**Cell 25 (Code):**
```python
# Visualization: Efficiency comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sig_level in enumerate(significance_levels):
    ax = axes[idx]
    sig_data = task3_df[task3_df['significance_level'] == sig_level]
    
    for model_name in models.keys():
        model_data = sig_data[sig_data['model'] == model_name]
        ax.scatter(
            model_data['standard_avg_set_size'],
            model_data['normalized_avg_set_size'],
            label=model_name,
            alpha=0.6
        )
    
    # Diagonal line (y=x)
    max_size = max(sig_data['standard_avg_set_size'].max(), sig_data['normalized_avg_set_size'].max())
    ax.plot([0, max_size], [0, max_size], 'r--', alpha=0.5, label='No improvement')
    ax.set_xlabel('Standard CP Avg Set Size')
    ax.set_ylabel('Normalized CP Avg Set Size')
    ax.set_title(f'Efficiency Comparison (ε={sig_level})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Cell 26 (Markdown):**
```markdown
### Task 3 Commentary

Normalized conformal classification uses difficulty estimation to create instance-specific predictions.
By grouping instances by difficulty (using entropy of predicted probabilities) and applying Mondrian
calibration, easier instances get smaller prediction sets while harder instances get larger sets.

**Real-world applications:**
- **Medical diagnosis**: Easy cases get confident single-class predictions, difficult cases get
  multiple possible diagnoses with guaranteed coverage
- **Quality control**: Clear-cut decisions for obvious cases, multiple options for borderline cases
- **Fraud detection**: High-confidence predictions for obvious fraud/legitimate, uncertainty sets
  for ambiguous cases

**Benefits:**
- More informative predictions for easier instances
- Maintains coverage guarantees
- Adapts to instance difficulty automatically
```

---

## Task 4: Interpretable Conformal Classifiers

### Add these cells after Task 3:

**Cell 27 (Markdown):**
```markdown
## Task 4: Interpretable Conformal Classification Trees

Conformal classification trees combine uncertainty quantification with interpretability.
Each leaf returns a conformal prediction set, and the tree structure provides clear
decision rules that are easy to understand and explain.
```

**Cell 28 (Code):**
```python
# Import Task 4 functions
from src.conformal.task4_interpretable import ConformalClassificationTree
```

**Cell 29 (Code):**
```python
# Evaluate conformal classification trees
task4_results = []

selected_datasets_task4 = dataset_files[:8]  # Use fewer datasets for trees (they're slower)

for dataset_name in tqdm(selected_datasets_task4, desc="Task 4: Datasets"):
    dataset_path = twoclass_dir / dataset_name
    
    try:
        X, y = load_dataset(dataset_path)
        
        if len(X) < 50 or len(y.unique()) != 2:  # Need more samples for trees
            continue
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Fewer folds
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            for sig_level in significance_levels:
                # Test with different tree depths
                for max_depth in [3, 5, 7]:
                    for mondrian in [True, False]:
                        try:
                            tree = ConformalClassificationTree(
                                max_depth=max_depth,
                                mondrian=mondrian,
                                significance_level=sig_level,
                                random_state=42
                            )
                            tree.fit(X_train_fold, y_train_fold)
                            
                            # Get predictions
                            pred_sets = tree.predict_conformal(X_test_fold)
                            point_preds = tree.predict(X_test_fold)
                            
                            # Evaluate
                            eval_results = evaluate_conformal_predictions(pred_sets, y_test_fold)
                            accuracy = accuracy_score(y_test_fold, point_preds)
                            
                            # Get tree info
                            leaf_info = tree.get_leaf_info()
                            
                            task4_results.append({
                                'dataset': dataset_name,
                                'fold': fold_idx,
                                'significance_level': sig_level,
                                'max_depth': max_depth,
                                'mondrian': mondrian,
                                'error_rate': eval_results['error_rate'],
                                'avg_set_size': eval_results['avg_set_size'],
                                'accuracy': accuracy,
                                'n_leaves': len(leaf_info),
                                'avg_leaf_samples': leaf_info['n_samples'].mean(),
                            })
                        except Exception as e:
                            continue
        
    except Exception as e:
        print(f"Error processing {dataset_name}: {str(e)}")
        continue

task4_df = pd.DataFrame(task4_results)
print(f"Task 4 results collected: {len(task4_df)} rows")
```

**Cell 30 (Code):**
```python
# Analyze conformal tree results
print("=" * 80)
print("TASK 4: INTERPRETABLE CONFORMAL TREES RESULTS")
print("=" * 80)

# Compare Mondrian vs Global
summary_task4 = task4_df.groupby(['mondrian', 'max_depth', 'significance_level']).agg({
    'error_rate': 'mean',
    'avg_set_size': 'mean',
    'accuracy': 'mean',
    'n_leaves': 'mean'
}).round(4)

print("\nComparison: Mondrian vs Global Calibration")
print(summary_task4)
```

**Cell 31 (Code):**
```python
# Visualize a conformal tree on one dataset
# Pick a dataset with reasonable size
example_dataset = None
for dataset_name in dataset_files[:10]:
    dataset_path = twoclass_dir / dataset_name
    try:
        X, y = load_dataset(dataset_path)
        if 50 <= len(X) <= 500 and len(y.unique()) == 2:
            example_dataset = (dataset_name, X, y)
            break
    except:
        continue

if example_dataset:
    dataset_name, X_ex, y_ex = example_dataset
    print(f"Visualizing tree for dataset: {dataset_name}")
    
    # Create and fit tree
    tree_viz = ConformalClassificationTree(
        max_depth=4,  # Shallow for interpretability
        mondrian=True,
        significance_level=0.05,
        random_state=42
    )
    tree_viz.fit(X_ex, y_ex)
    
    # Visualize
    feature_names = [f'Feature_{i}' for i in range(X_ex.shape[1])]
    class_names = [f'Class_{i}' for i in range(len(y_ex.unique()))]
    tree_viz.visualize_tree(feature_names=feature_names[:10], class_names=class_names)  # Limit features for readability
    
    # Show leaf information
    print("\nLeaf Information:")
    print(tree_viz.get_leaf_info())
else:
    print("No suitable dataset found for visualization")
```

**Cell 32 (Code):**
```python
# Visualization: Tree depth vs efficiency
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sig_level in enumerate(significance_levels):
    ax = axes[idx]
    sig_data = task4_df[task4_df['significance_level'] == sig_level]
    
    for mondrian in [True, False]:
        mondrian_data = sig_data[sig_data['mondrian'] == mondrian]
        depths = sorted(mondrian_data['max_depth'].unique())
        avg_sizes = [mondrian_data[mondrian_data['max_depth']==d]['avg_set_size'].mean() for d in depths]
        
        label = 'Mondrian' if mondrian else 'Global'
        ax.plot(depths, avg_sizes, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Tree Depth')
    ax.set_ylabel('Average Set Size')
    ax.set_title(f'Efficiency vs Tree Depth (ε={sig_level})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Cell 33 (Markdown):**
```markdown
### Task 4 Commentary

Conformal classification trees provide an interpretable way to combine uncertainty quantification
with explainability. The decision tree structure makes it easy to understand why certain predictions
are made, while conformal prediction provides coverage guarantees.

**Key observations:**
- **Shallow trees** (depth 3-5) are more interpretable and allow Mondrian calibration per leaf
- **Deeper trees** may have too many leaves for effective Mondrian calibration
- **Mondrian calibration** provides local guarantees per leaf, potentially better efficiency
- **Global calibration** is simpler but may be less efficient

**Trade-offs:**
- Interpretability vs. predictive performance
- Tree depth vs. number of leaves (affects Mondrian calibration)
- Coverage guarantees are maintained regardless of calibration method

**Applications:**
- Decision support systems where explanations are required
- Regulatory compliance (explainable AI)
- Educational tools for understanding model behavior
```

---

## Summary

After adding all these cells, your notebook will have:

1. **Task 1**: Complete (already done)
2. **Task 2**: Class-conditional conformal prediction evaluation with class-specific error rate comparison
3. **Task 3**: Normalized conformal classification with difficulty-based Mondrian categories
4. **Task 4**: Interpretable conformal classification trees with visualization

All Python code is in the `src/conformal/` package, and you just need to add the notebook cells as shown above.

