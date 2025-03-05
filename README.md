## **Automated Machine Learning Benchmarking Library**
🚀 **AutoMLBench** provides a seamless way to compare machine learning models, preprocess data, evaluate performance, and optimize models with **hyperparameter tuning**.

---

## **📌 Installation**
Ensure all dependencies are installed:
```bash
pip install pandas scikit-learn numpy matplotlib xgboost lightgbm catboost imbalanced-learn
```
Install from pypi:
```bash
pip install automlbench
```

For local development:
```bash
git clone https://github.com/AnnNaserNabil/automlbench.git
cd automlbench
pip install -e .
```

## **Model Comparison Without Hyperparameter Tuning**
The simplest way to compare models using **AutoMLBench**.

### **1️⃣ Load Dataset & Preprocess**
```python
import pandas as pd
from automlbench import preprocess_data, get_models, train_models, evaluate_model, plot_performance

# Load dataset
url = "DATAPATH"
df = pd.read_csv(url) 

# Define target column
target_column = "Name OF the Target Column"

# Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
```

---

### **2️⃣ Train All Default Models**
```python
# Get predefined models
models = get_models()

# Train models without tuning
results = train_models(X_train, X_test, y_train, y_test)

# Print model performance results
print(results)
```

---

### **3️⃣ Evaluate & Compare Model Performance**
```python
# Evaluate all models
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model.fit(X_train, y_train), X_test, y_test)
    print(metrics)

# Plot performance comparison
plot_performance(results, metrics=["Accuracy", "Precision", "Recall", "F1-Score", "RMSE"])
```

---

## **🔹 Model Comparison With Hyperparameter Tuning**
For better performance, use **hyperparameter tuning**.

### **1️⃣ Get Hyperparameter Grids**
```python
from automlbench import get_hyperparameter_grids, tune_hyperparameters

# Retrieve hyperparameter grids
hyperparameter_grids = get_hyperparameter_grids()
```

---

### **2️⃣ Tune Models**
```python
best_models = {}

# Tune each model if it has a predefined hyperparameter grid
for model_name, model in models.items():
    if model_name in hyperparameter_grids:
        print(f"Tuning {model_name}...")
        best_model, best_params = tune_hyperparameters(model, hyperparameter_grids[model_name], X_train, y_train)
        best_models[model_name] = best_model
        print(f"Best params for {model_name}: {best_params}")
    else:
        best_models[model_name] = model  # Use default if no tuning grid
```

---

### **3️⃣ Train Tuned Models**
```python
# Train models using the best hyperparameters found
tuned_results = train_models(
    X_train, X_test, y_train, y_test, 
    selected_models=list(best_models.keys()), 
    hyperparams={name: model.get_params() for name, model in best_models.items()}
)

# Display tuned model results
print(tuned_results)
```

---

### **4️⃣ Evaluate & Compare Tuned Models**
```python
# Evaluate all tuned models
for model_name, model in best_models.items():
    print(f"Evaluating {model_name}...")
    metrics = evaluate_model(model.fit(X_train, y_train), X_test, y_test)
    print(metrics)

# Plot comparison of tuned models
plot_performance(tuned_results, metrics=["Accuracy", "Precision", "Recall", "F1-Score", "RMSE"])
```


## **⚡ Quick Summary**
✅ **Basic Comparison** – Train models with default settings.  
✅ **Hyperparameter Tuning** – Optimize models for better performance.  
✅ **Evaluation & Visualization** – Compare accuracy, precision, recall, F1-score, and RMSE.  
✅ **Automated ML Benchmarking** – Quickly assess multiple models with minimal code.  

---

## **📌 Contributing**
Contributions are welcome! To contribute:
1. **Fork the repository**
2. **Create a new branch (`feature-branch`)**
3. **Make changes & test (`pytest tests/`)**
4. **Submit a pull request (PR)**

---

## **📜 License**
AutoMLBench is released under the **MIT License**.

---

This **documentation makes it easy** for users to **compare models before and after tuning** using **AutoMLBench**. 🚀 Let me know if you need modifications! 🔥
