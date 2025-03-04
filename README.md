
### **Automated Machine Learning Benchmarking Library**
📌 **AutoMLBench** is a Python library designed to **automate the machine learning pipeline**, including:
- **Data loading**
- **Preprocessing**
- **Model training**
- **Evaluation**
- **Performance visualization**

---

## **Installation**
Ensure you have all the necessary dependencies installed:
```bash
pip install pandas scikit-learn numpy matplotlib xgboost lightgbm catboost imbalanced-learn
```

For local development, clone the repository and install it in **editable mode**:
```bash
git clone https://github.com/AnnNaserNabil/automlbench.git
cd automlbench
pip install -e .
```
Install from pypi
```bash
pip install automlbench
```

---

## **Modules Overview**
AutoMLBench consists of several modules, each handling a specific part of the ML pipeline.

| Module                | Functionality |
|-----------------------|--------------|
| `data_loader.py`      | Loads data from multiple file formats (`CSV`, `Excel`, `JSON`, `Parquet`, `HDF5`). |
| `preprocessing.py`    | Handles missing values, feature scaling, and categorical encoding. |
| `models.py`          | Provides predefined machine learning models (Random Forest, XGBoost, LightGBM, etc.). |
| `model_train.py`      | Trains multiple models with class balancing and metric evaluation. |
| `hyperparameter_tuning.py` | Uses `GridSearchCV` for hyperparameter optimization. |
| `evaluation.py`       | Computes performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC). |
| `visualization.py`    | Generates performance comparison plots. |
| `utils.py`            | Provides logging and execution time utilities. |
| `__init__.py`         | Exposes core functionalities for easy import. |

---

## **Usage Guide**
### **1️⃣ Load the Dataset**
AutoMLBench supports direct loading of datasets.
```python
import pandas as pd
from automlbench import load_data

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
```

---

### **2️⃣ Preprocess the Data**
```python
from automlbench import preprocess_data

# Define the target column
target_column = "Survived"

# Preprocess dataset (handles missing values, encoding, scaling)
X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
```

---

### **3️⃣ Train Machine Learning Models**
```python
from automlbench import get_models, train_models

# Get predefined models
models = get_models()

# Train all models
results = train_models(X_train, X_test, y_train, y_test)

# Display model results
print(results)
```

---

### **4️⃣ Evaluate Model Performance**
```python
from automlbench import evaluate_model

# Evaluate a specific model (e.g., Random Forest)
rf_model = models["Random Forest"].fit(X_train, y_train)
metrics = evaluate_model(rf_model, X_test, y_test)

print(metrics)
```

---

### **5️⃣ Visualize Model Performance**
```python
from automlbench import plot_performance

# Plot model comparison for multiple metrics
plot_performance(results, metrics=["Accuracy", "Precision", "Recall", "F1-Score", "RMSE"])
```

---

### **6️⃣ Hyperparameter Tuning (Optional)**
If you want to fine-tune a model:
```python
from automlbench import tune_hyperparameters

# Define hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20]
}

# Tune the Random Forest model
best_model, best_params = tune_hyperparameters(models["Random Forest"], param_grid, X_train, y_train)

print(f"Best Model: {best_model}")
print(f"Best Parameters: {best_params}")
```

---

## **Example Workflow**
Here's a full **end-to-end pipeline** using AutoMLBench:
```python
import pandas as pd
from automlbench import (
    preprocess_data, get_models, train_models, evaluate_model, plot_performance
)

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df, "Survived")

# Train models
results = train_models(X_train, X_test, y_train, y_test)

# Display evaluation metrics
for model_name, model in get_models().items():
    model.fit(X_train, y_train)
    print(f"{model_name} Metrics:", evaluate_model(model, X_test, y_test))

# Plot performance
plot_performance(results)
```

---

## **Troubleshooting**
### **Common Issues & Fixes**
#### ❌ **ImportError: cannot import name 'train_models'**
✔ **Fix:** Ensure `train_models` is listed in `__init__.py`:
```python
from .model_train import train_models
```

#### ❌ **ModuleNotFoundError: No module named 'automlbench'**
✔ **Fix:** Reinstall the package in editable mode:
```bash
pip install -e .
```

#### ❌ **ValueError: The target variable must contain at least two classes**
✔ **Fix:** Ensure the dataset has at least **two unique classes** in the target column.

---

## **Future Improvements**
✅ **Ensemble Model Support**  
✅ **Feature Selection Methods**  
✅ **AutoML Integration** (e.g., with Optuna, Hyperopt)  
✅ **Support for Regression Models**  

---

## **Contributing**
We welcome contributions! To contribute:
1. **Fork the repository**
2. **Create a new branch** (`feature-branch`)
3. **Make changes and test**
4. **Submit a pull request (PR)**

---

## **License**
AutoMLBench is released under the **MIT License**.
