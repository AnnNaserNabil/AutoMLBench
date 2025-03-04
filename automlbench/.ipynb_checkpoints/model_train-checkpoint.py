from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
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

def train_models(X_train, y_train, X_test, y_test, selected_models=None, hyperparams=None, metrics=None):
    """Trains and evaluates multiple models with user control."""

    # Default models
    models = {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=min(3, len(X_train))),  # Ensure n_neighbors <= n_samples
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    # Filter selected models if provided
    if selected_models:
        models = {name: models[name] for name in selected_models if name in models}

    # Apply hyperparameters if provided
    if hyperparams:
        for model_name, params in hyperparams.items():
            if model_name in models:
                models[model_name].set_params(**params)

    # Default evaluation metrics
    default_metrics = {
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1-Score": f1_score,
        "RMSE": lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False)
    }

    # Use user-defined metrics if provided
    if metrics:
        evaluation_metrics = {name: default_metrics[name] for name in metrics if name in default_metrics}
    else:
        evaluation_metrics = {"Accuracy": accuracy_score}

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute all selected metrics
        results[name] = {metric_name: metric_func(y_test, y_pred) for metric_name, metric_func in evaluation_metrics.items()}

    return results
