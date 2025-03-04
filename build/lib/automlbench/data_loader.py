import pandas as pd

def load_data(filepath_or_df):
    """Loads dataset from a file or DataFrame."""
    if isinstance(filepath_or_df, pd.DataFrame):
        return filepath_or_df
    return pd.read_csv(filepath_or_df) if filepath_or_df.endswith('.csv') else pd.read_excel(filepath_or_df)
