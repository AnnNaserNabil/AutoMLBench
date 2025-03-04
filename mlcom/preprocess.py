import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np

def preprocess_data(df, target_column, auto=True, manual_features=None, scaling_method='standard', encoding_method='label'):
    """Handles automatic and manual feature engineering."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # If manual features are provided, return them directly
    if manual_features is not None:
        return manual_features, y

    if auto:
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        # Encoding categorical variables
        if encoding_method == 'label':
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col])
        elif encoding_method == 'onehot':
            X = pd.get_dummies(X, columns=categorical_cols)

        # Scaling numerical variables
        if scaling_method == 'standard':
            X[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])
        elif scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            X[numerical_cols] = MinMaxScaler().fit_transform(X[numerical_cols])

    return X, y
