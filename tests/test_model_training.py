import pandas as pd
import numpy as np
from automlbench import get_models, train_models


def test_train_models():
    # Create synthetic dataset
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(20, 5)
    y_train = np.random.choice([0, 1], size=100)
    y_test = np.random.choice([0, 1], size=20)

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Ensure results are generated
    assert isinstance(results, dict)
    assert "Accuracy" in results[list(results.keys())[0]]  # Check accuracy is included
