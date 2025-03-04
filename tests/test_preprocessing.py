import pandas as pd
import numpy as np
from automlbench import preprocess_data


def test_preprocess_data():
    # Create a mock dataset
    data = pd.DataFrame(
        {
            "Feature1": [1, 2, np.nan, 4, 5],
            "Feature2": ["A", "B", "A", "B", "C"],
            "Target": [0, 1, 0, 1, 0],
        }
    )

    # Run preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(data, "Target")

    # Ensure no missing values
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()

    # Check label encoding
    assert set(y_train) == {0, 1}
    assert set(y_test) == {0, 1}
