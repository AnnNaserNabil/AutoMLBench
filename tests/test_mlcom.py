import pandas as pd
from mlcom import load_data, preprocess_data, train_models

def test_load_data():
    df = load_data("tests/sample.csv")
    assert isinstance(df, pd.DataFrame)

def test_preprocess_data():
    df = pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]})
    X, y = preprocess_data(df, target_column="label")
    assert len(X) == len(y)

def test_train_models():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "label": [0, 1, 0, 1]})
    X, y = preprocess_data(df, target_column="label")
    results = train_models(X, y, X, y)
    assert isinstance(results, dict)
