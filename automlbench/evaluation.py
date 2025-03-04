### evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
        "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=1),
    }
    
    if hasattr(model, "predict_proba"):
        if len(set(y_test)) > 2:  # Multiclass case
            try:
                metrics["AUC-ROC"] = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            except ValueError:
                metrics["AUC-ROC"] = np.nan  # Handle cases where AUC-ROC cannot be computed
        else:  # Binary case
            metrics["AUC-ROC"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    return metrics