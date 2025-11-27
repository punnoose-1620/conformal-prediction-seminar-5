# State-of-the-Art in AI Research - [Seminar 5](Assignment.md)

This repository contains two main projects:

1. **Agent-Based SIR Epidemic Simulation with Reinforcement Learning** - An implementation of an agent-based SIR epidemic model with RL for optimal intervention policy discovery.

2. **Conformal Prediction for Classification** - Complete implementation of conformal classification methods including standard, class-conditional, normalized, and interpretable variants.

---

# Part 1: Agent-Based SIR Epidemic Simulation with Reinforcement Learning

A Python implementation of an agent-based Susceptible-Infectious-Recovered (SIR) epidemic model with reinforcement learning for optimal intervention policy discovery. This project demonstrates how Q-learning can be used to learn adaptive intervention strategies that balance epidemiological outcomes with social costs.

## Overview

This project implements a well-mixed agent-based SIR epidemic simulator where:
- Each agent can be in one of three states: **Susceptible (S)**, **Infectious (I)**, or **Recovered (R)**
- Infectious agents make contacts with other agents, potentially transmitting the disease
- A reinforcement learning agent learns optimal intervention policies to minimize total cost (epidemiological burden + social intervention costs)
- Empirical methods are provided to estimate the basic reproduction number (R₀)

## Features

### Core Components

1. **Agent-Based SIR Simulator** ([`sir_sim.py`](./src/sir_rl/sir_sim.py))
   - Well-mixed population dynamics
   - Per-contact transmission probability: `p_trans = 1 - exp(-β·dt)`
   - Per-step recovery probability: `p_rec = 1 - exp(-γ·dt)`
   - Configurable population size, initial conditions, and epidemic parameters

2. **Intervention Framework** ([`interventions.py`](./src/sir_rl/interventions.py))
   - Two intervention types:
     - **Contact reduction**: Reduces the number of contacts per infectious agent
     - **Transmission reduction**: Reduces the transmission rate β
   - Cost function balancing epidemiological impact and social costs

3. **Reinforcement Learning Agent** ([`rl_agent.py`](./src/sir_rl/rl_agent.py))
   - Tabular Q-learning implementation
   - State discretization based on susceptible/infectious fractions and time
   - Action space: intervention levels [0.0, 0.25, 0.5, 0.75, 1.0]
   - Epsilon-greedy exploration with decay

4. **Empirical R₀ Estimation** ([`experiments.py`](./src/sir_rl/experiments.py))
   - Index-case method for estimating basic reproduction number
   - Beta parameter sweep functionality
   - Statistical aggregation across multiple runs

5. **Visualization Utilities** ([`utils.py`](./src/sir_rl/utils.py))
   - SIR dynamics plotting
   - Training curve visualization with moving averages

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Setup

1. Clone or download this repository:
```bash
cd project-sir-rl
```

2. Create a virtual environment (recommended):
```bash
# Using venv
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Or using conda
conda env create -f environment.yml
conda activate sir_rl
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project-root/
├── README.md                 # This file
├── Assignment.md             # Conformal prediction assignment description
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment (optional)
├── notebooks/
│   ├── ABS_RL.ipynb        # SIR-RL experiment notebook
│   └── Conformal_Classification.ipynb  # Conformal prediction notebook
├── twoclass/                # Two-class datasets for conformal prediction
├── data/                    # Optional datasets
├── src/
│   ├── sir_rl/             # SIR-RL Python package
│   │   ├── __init__.py
│   │   ├── config.py       # Default parameters
│   │   ├── sir_sim.py      # Core SIR simulator
│   │   ├── interventions.py # Intervention logic
│   │   ├── rl_agent.py     # Q-learning agent
│   │   ├── experiments.py # R₀ estimation
│   │   └── utils.py        # Plotting utilities
│   └── conformal/          # Conformal prediction package
│       ├── __init__.py
│       ├── task2_evaluation.py  # Class-conditional CP evaluation
│       ├── task3_normalized.py   # Normalized CP implementation
│       └── task4_interpretable.py # Interpretable CP trees
├── tests/
│   └── test_sir.py         # Unit tests
└── results/                 # Output plots and data
```

## Usage

### Quick Start

Open the Jupyter notebook to run experiments:

```bash
jupyter lab notebooks/ABS_RL.ipynb
```

Or run from Python:

```python
from src.sir_rl.sir_sim import run_sim
from src.sir_rl.rl_agent import train_q_learning
from src.sir_rl.utils import plot_sir, plot_training

# Run a basic simulation
data = run_sim(N=2000, I0=5, beta=0.15, gamma=1/7, C=8, dt=1.0, T=160, seed=42)
plot_sir(data, title="SIR Dynamics (N=2000)")

# Train Q-learning agent
train_res = train_q_learning(
    N=1000, I0=5, beta=0.15, gamma=1/7, C=8, dt=1.0, T=120,
    n_episodes=300, n_bins=6, t_bins=8
)
plot_training(train_res["history"], window=20, title="Q-Learning Training")
```

### Running Tests

```bash
pytest tests/
```

### Key Parameters

- `N`: Population size (default: 1000)
- `I0`: Initial number of infectious agents (default: 5)
- `beta`: Transmission rate (default: 0.15)
- `gamma`: Recovery rate per day (default: 1/7)
- `C`: Contacts per infectious agent per step (default: 8)
- `dt`: Time step size (default: 1.0)
- `T`: Simulation horizon (default: 150)
- `lambda_epi`: Epidemiological cost weight (default: 1.0)
- `lambda_soc`: Social intervention cost weight (default: 0.1)

## Reproducibility

- **Python Version**: 3.10+
- **Random Seed**: 42 (configurable in [`config.py`](./src/sir_rl/config.py) and notebook)
- **Dependencies**: Pinned in [`requirements.txt`](./requirements.txt)
- **Reproduction Steps**:
  1. Create environment and install requirements
  2. Open [`notebooks/ABS_RL.ipynb`](./notebooks/ABS_RL.ipynb)
  3. Run cells sequentially from top to bottom
  4. For experimental sweeps, use fixed seed offsets and average across independent runs

## Example Workflow

1. **Baseline Simulation**: Run SIR simulation without interventions
2. **RL Training**: Train Q-learning agent to learn optimal intervention policy
3. **Policy Evaluation**: Compare learned policy against baselines (no intervention, constant intervention)
4. **R₀ Estimation**: Estimate basic reproduction number for different parameter values

## Mathematical Background

### SIR Dynamics

The agent-based model implements:
- **Transmission**: Each infectious agent contacts up to `C` others per time step
- **Infection probability**: `p_trans = 1 - exp(-β·dt)` per contact with a susceptible
- **Recovery probability**: `p_rec = 1 - exp(-γ·dt)` per time step

### Cost Function

The per-step cost balances epidemiological and social factors:
```
cost = λ_epi · new_infections + λ_soc · u²
```
where `u ∈ [0,1]` is the intervention level.

### Q-Learning

The agent learns a value function `Q(s,a)` using:
```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```
where states are discretized based on (S/N, I/N, t/T).

## Contributing

This project follows a modular design:
- Core simulation in `sir_sim.py`
- Interventions in `interventions.py`
- Learning in `rl_agent.py`
- Experiments in `experiments.py`
- Utilities in `utils.py`

This structure makes it easy to extend the model (e.g., network-based contact structures, different intervention types, alternative RL algorithms).

## License

See LICENSE file for details.

## References

- SIR model: Kermack & McKendrick (1927)
- Q-learning: Watkins & Dayan (1992)

## Future Extensions

Potential enhancements:
- Network-based contact structures (small-world, scale-free)
- Multi-objective optimization
- Deep Q-Networks (DQN) for larger state spaces
- Different intervention types (testing, isolation, vaccination)
- Age-structured or spatially-explicit models

---

# Part 2: Conformal Prediction for Classification

This project implements conformal classification methods for providing prediction sets with guaranteed coverage properties. All tasks from the assignment ([`Assignment.md`](Assignment.md)) have been completed.

## Completed Tasks

### ✅ Task 1: Empirical Investigation of Conformal Classification (Mandatory)

**Status**: Complete

**Implementation**: 
- Inductive (split) conformal classification implemented in [`notebooks/Conformal_Classification.ipynb`](./notebooks/Conformal_Classification.ipynb)
- Evaluation across 10+ two-class datasets from the [`twoclass/`](./twoclass/) folder
- Three base models: Random Forest, Decision Tree, and XGBoost
- 10-fold cross-validation
- Three significance levels tested: ε = 0.05, 0.10, 0.20 (95%, 90%, 80% confidence)

**Results Reported**:
- Empirical error rates (validating coverage guarantees)
- Average prediction set size (efficiency metric)
- Singleton rate (fraction of single-class predictions)
- Standard accuracy and AUC for comparison

**Notebook Contents**:
- Description of conformal classification procedure, guarantees, and informal reasoning
- Well-commented implementation code
- Results visualization and commentary

### ✅ Task 2: Class-Conditional Conformal Prediction (Optional)

**Status**: Complete

**Implementation**: 
- Class-conditional conformal prediction (CCCP) using Mondrian approach
- Module: [`src/conformal/task2_evaluation.py`](./src/conformal/task2_evaluation.py)
- Provides independent coverage guarantees for each class

**Features**:
- Class-specific threshold computation
- Comparison with standard conformal prediction
- Class-specific error rate reporting
- Visualization of error rate imbalances

**Results**: 
- Demonstrates how CCCP addresses imbalanced error rates across classes
- Shows more balanced error rates compared to standard CP

### ✅ Task 3: Normalized Conformal Classification (Optional)

**Status**: Complete

**Implementation**:
- Difficulty-based Mondrian categories using entropy estimation
- Module: [`src/conformal/task3_normalized.py`](./src/conformal/task3_normalized.py)
- Instance-specific predictions based on difficulty

**Features**:
- Difficulty estimation using entropy of predicted probabilities
- Quantile-based difficulty binning
- Category-specific thresholds for more efficient predictions
- More specific predictions for easier instances

**Real-World Applications** (documented in notebook):
- Medical diagnosis: confident predictions for clear cases, uncertainty sets for difficult cases
- Quality control: clear-cut decisions vs. multiple options for borderline cases
- Fraud detection: high-confidence for obvious cases, uncertainty for ambiguous cases

### ✅ Task 4: Interpretable Conformal Classification Trees (Optional)

**Status**: Complete

**Implementation**:
- Conformal classification trees with interpretable decision structure
- Module: [`src/conformal/task4_interpretable.py`](./src/conformal/task4_interpretable.py)
- Class: `ConformalClassificationTree`

**Features**:
- Decision trees where each leaf returns a conformal prediction set
- Support for both Mondrian (leaf-specific) and global calibration
- Tree visualization with `visualize_tree()` method
- Leaf information extraction
- Configurable tree depth for interpretability vs. performance trade-off

**Key Capabilities**:
- Automatic validation of feature names
- Leaf-specific threshold information
- Comparison of Mondrian vs. global calibration
- Efficiency analysis across different tree depths

## Conformal Prediction Usage

### Running the Notebook

```bash
jupyter lab notebooks/Conformal_Classification.ipynb
```

The notebook contains:
- **Task 1**: Complete empirical evaluation (Cells 0-11)
- **Task 2**: Class-conditional comparison (Cells 15-20)
- **Task 3**: Normalized conformal classification (Cells 21-26)
- **Task 4**: Interpretable trees (Cells 27-33)

### Using the Python Modules

```python
# Task 2: Class-conditional evaluation
from src.conformal.task2_evaluation import evaluate_class_conditional_comparison

# Task 3: Normalized conformal classification
from src.conformal.task3_normalized import normalized_conformal_classification

# Task 4: Interpretable trees
from src.conformal.task4_interpretable import ConformalClassificationTree

tree = ConformalClassificationTree(max_depth=5, mondrian=True, significance_level=0.05)
tree.fit(X_train, y_train)
pred_sets = tree.predict_conformal(X_test)
tree.visualize_tree()
```

## Conformal Prediction Key Concepts

### Coverage Guarantees

All methods provide marginal coverage guarantees:
- **Standard CP**: Global guarantee that true class is in prediction set with probability ≥ 1-ε
- **Class-Conditional CP**: Independent guarantee per class
- **Normalized CP**: Guarantee per difficulty category
- **Interpretable Trees**: Global or leaf-specific guarantees depending on calibration

### Efficiency Metrics

- **Average set size**: Smaller sets are more informative
- **Singleton rate**: Fraction of single-class predictions
- **Difficulty-specific metrics**: For normalized CP, efficiency varies by instance difficulty

## Dependencies for Conformal Prediction

The conformal prediction tasks require:
- `scikit-learn` (for base classifiers and tree visualization)
- `pandas` (for data handling)
- `xgboost` (optional, for XGBoost classifier)
- `matplotlib` (for visualizations)
- `numpy` (for numerical operations)

All dependencies are included in [`requirements.txt`](./requirements.txt).

## References

- **Conformal Prediction**: Vovk, Gammerman, and Shafer (2005)
- **Mondrian Conformal Prediction**: Vovk, Nouretdinov, and Gammerman (2005)
- **Assignment Details**: See [`Assignment.md`](./Assignment.md)

