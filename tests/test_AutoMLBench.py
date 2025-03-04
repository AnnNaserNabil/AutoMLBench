import pandas as pd
from AutoMLBench import load_data, preprocess_data, train_models

def test_load_data():
    df = load_data("tests/sample.csv")
    assert isinstance(df, pd.DataFrame)

def test_preprocess_data():
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})
    X, y = preprocess_data(df, target_column="label")
    assert len(X) == len(y)

def test_train_models():
    df = pd.DataFrame({"feature": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 0, 1]})
    X, y = preprocess_data(df, target_column="label")
    results = train_models(X, y, X, y)
    assert isinstance(results, dict)
