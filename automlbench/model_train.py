# model_train.py

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

def train_models(X_train, X_test, y_train, y_test, selected_models=None, hyperparams=None):
    """
    Trains multiple machine learning models and evaluates their performance.

    Parameters:
    - X_train (pd.DataFrame): Training feature matrix.
    - X_test (pd.DataFrame): Test feature matrix.
    - y_train (pd.Series): Training target variable.
    - y_test (pd.Series): Test target variable.
    - selected_models (list, optional): Names of models to train (default: all models).
    - hyperparams (dict, optional): Custom hyperparameters for models.

    Returns:
    - dict: Model performance results.
    """

    # ✅ Step 1: Ensure `y_train` and `y_test` have both classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Error: The target variable must contain at least two classes in both training and test sets.")

    # ✅ Step 2: Handle class imbalance **only on training data**
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # ✅ Step 3: Define available models
    models = {
        "Random Forest": RandomForestClassifier(class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(class_weight="balanced"),
        "AdaBoost": AdaBoostClassifier(),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
        "Logistic Regression": LogisticRegression(class_weight="balanced"),
        "Support Vector Machine": SVC(class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    # ✅ Step 4: Filter models if specific ones are selected
    if selected_models:
        models = {name: models[name] for name in selected_models if name in models}

    # ✅ Step 5: Apply hyperparameters if provided
    if hyperparams:
        for model_name, params in hyperparams.items():
            if model_name in models:
                models[model_name].set_params(**params)

    # ✅ Step 6: Define evaluation metrics
    evaluation_metrics = {
        "Accuracy": accuracy_score,
        "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted", zero_division=1),
        "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted", zero_division=1),
        "F1-Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted", zero_division=1),
        "RMSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
    }

    results = {}

    # ✅ Step 7: Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        # ✅ Step 8: Compute and store performance metrics
        results[name] = {metric_name: metric_func(y_test, y_pred) for metric_name, metric_func in evaluation_metrics.items()}

        # ✅ Step 9: Debug - Check unique predictions
        print(f"{name}: Unique Predictions - {np.unique(y_pred)}")

    return results
