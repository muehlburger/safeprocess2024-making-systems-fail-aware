import pandas as pd
from typing import Dict
from sklearn.preprocessing import MinMaxScaler
import logging
import tensorflow as tf
import numpy as np

log = logging.getLogger(__name__)

def fit_scaler(data: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data.values)
    return scaler

def tranform_scaler(scaler: MinMaxScaler, data: pd.DataFrame) -> pd.DataFrame:
    scaled_data = scaler.transform(data.values)
    return pd.DataFrame(scaled_data, index=data.index, columns=data.columns)

def process_data(data: pd.DataFrame, parameters: Dict):
    data = data.dropna()
    print(data.columns)
    data = data[parameters["feature"]].copy()
    #time = np.arange(len(data), dtype="float32")
    #values = data[parameters["feature"]].values

    return data

def create_model_input_table(data: pd.DataFrame, parameters: Dict):
    """Combines all data to create a model input table.

    Args:
        healthy: Preprocessed data for healthy signals.
    Returns:
        model input table.
    """
    train_x = data[parameters["feature"]]
    return pd.DataFrame(train_x, index=data.index, columns=data.columns)
