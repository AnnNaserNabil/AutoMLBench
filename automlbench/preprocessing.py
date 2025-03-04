### preprocessing.py
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold


def preprocess_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Normalize class labels to ensure they start from 0 and are continuous
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    num_features = X.select_dtypes(include=[np.number]).columns
    cat_features = X.select_dtypes(exclude=[np.number]).columns

    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),  # Convert to dense
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)]
    )

    X_preprocessed = preprocessor.fit_transform(X)
    X_preprocessed = np.asarray(X_preprocessed)  # Ensure dense output

    # Use stratified split to preserve class distribution
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(X_preprocessed, y):
        X_train, X_test = X_preprocessed[train_index], X_preprocessed[test_index]
        y_train, y_test = y[train_index], y[test_index]
        break  # Use only the first split

    return X_train, X_test, y_train, y_test
