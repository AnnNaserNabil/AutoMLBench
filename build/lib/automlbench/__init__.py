### __init__.py
from .data_loader import load_data
from .preprocessing import preprocess_data
from .models import get_models, get_hyperparameter_grids
from .evaluation import evaluate_model
from .visualization import plot_performance
from .hyperparameter_tuning import tune_hyperparameters
from .utils import time_execution, log_message
from .model_train import train_models


import os
import warnings

def suppress_warnings(suppress=True):
    if suppress:
        warnings.filterwarnings("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"
    else:
        warnings.resetwarnings()
        os.environ.pop("PYTHONWARNINGS", None)

__all__ = [
    "load_data",
    "preprocess_data",
    "get_models",
    "get_hyperparameter_grids",
    "evaluate_model",
    "plot_performance",
    "tune_hyperparameters",
    "time_execution",
    "log_message",
    "suppress_warnings",
    "train_models"  
]

