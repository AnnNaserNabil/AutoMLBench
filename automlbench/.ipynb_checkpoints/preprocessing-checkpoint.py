import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
import numpy as np

def preprocess_data(df, target_column, auto=True, manual_features=None, scaling_method='standard', encoding_method='label', handle_missing='mean'):
    """Handles automatic and manual feature engineering together."""

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Apply manual feature transformations first
    if manual_features is not None:
        for col, transformation in manual_features.items():
            X[col] = transformation(X)

    if auto:
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        # Handling missing values
        if handle_missing == 'mean':
            X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())
        elif handle_missing == 'median':
            X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())
        elif handle_missing == 'mode':
            X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

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
            X[numerical_cols] = MinMaxScaler().fit_transform(X[numerical_cols])

    return X, y
