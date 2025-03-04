### models.py
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

def get_models():
    return {
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=500),  # Increased max_iter
        "Support Vector Machine": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(max_iter=500),  # Increased max_iter
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

def get_hyperparameter_grids():
    return {
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "Extra Trees": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
        "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "Decision Tree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
        "Logistic Regression": {"C": [0.1, 1, 10]},
        "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "K-Nearest Neighbors": {"n_neighbors": [3, 5, 10]},
        "Neural Network": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.01]},
        "XGBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "LightGBM": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
        "CatBoost": {"iterations": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
    }
