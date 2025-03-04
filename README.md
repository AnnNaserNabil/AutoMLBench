# AutoMLBench- Automated ML Model Benchmarking Library

Automlbench is a Python library for automated machine learning model benchmarking. It simplifies the process of comparing multiple machine learning models by providing utilities for data loading, preprocessing, model selection, hyperparameter tuning, evaluation, and visualization. The library is designed to streamline model experimentation and performance analysis, making it ideal for data scientists and machine learning practitioners.

## 🚀 Features
✅ **Automated model benchmarking** – Compare multiple models with minimal effort.  
✅ **Flexible preprocessing** – Choose between automatic or manual feature engineering.  
✅ **Performance visualization** – Generate insightful plots for model comparison.  
✅ **Customizable feature handling** – Supports missing value imputation, scaling, and encoding.  
✅ **Multi-model training** – Supports **Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost**, and more.



## Installation
```bash
pip install automlbench
```

## Importing Automlbench
```python
from automlbench import (
    load_data, preprocess_data, get_models, get_hyperparameter_grids,
    evaluate_model, plot_performance, tune_hyperparameters, 
    time_execution, log_message, suppress_warnings
)
```

## Features
### 1. Load Data
```python
df = load_data("dataset.csv")
```

### 2. Preprocess Data
```python
X_train, X_test, y_train, y_test = preprocess_data(df, target_column="target")
```

### 3. Get Available Models
```python
models = get_models()
```

### 4. Hyperparameter Grids
```python
param_grids = get_hyperparameter_grids()
```

### 5. Evaluate Models
```python
results = {name: evaluate_model(model, X_test, y_test) for name, model in trained_models.items()}
```

### 6. Hyperparameter Tuning
```python
best_models = tune_hyperparameters(models, X_train, y_train, param_grids)
```

### 7. Plot Performance
```python
plot_performance(results)
```

### 8. Suppress Warnings (Optional)
```python
suppress_warnings(True)  # Set to False if you want to see warnings
```

## Utilities
- `time_execution(func)`: Measure execution time of a function.
- `log_message(msg)`: Log messages for debugging.

## License
This project is licensed under the MIT License.
