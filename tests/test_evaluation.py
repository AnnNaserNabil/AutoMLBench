import numpy as np
from sklearn.ensemble import RandomForestClassifier
from automlbench import evaluate_model

def test_evaluate_model():
    # Create dummy dataset
    X_test = np.random.rand(20, 5)
    y_test = np.random.choice([0, 1], size=20)

    # Train model
    model = RandomForestClassifier().fit(X_test, y_test)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Ensure all expected metrics exist
    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1 Score" in metrics
