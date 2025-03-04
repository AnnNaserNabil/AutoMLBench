### data_loader.py
import os
import pandas as pd

def load_data(data_path):
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(data_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(data_path)
    elif ext == ".json":
        return pd.read_json(data_path)
    elif ext == ".parquet":
        return pd.read_parquet(data_path)
    elif ext == ".hdf5":
        return pd.read_hdf(data_path)
    else:
        raise ValueError("Unsupported file format")