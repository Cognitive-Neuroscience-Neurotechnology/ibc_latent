import numpy as np
import pandas as pd

def normalize_data(data):
    """Normalize the data to have zero mean and unit variance."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / std
    return normalized_data

def reshape_data(data, new_shape):
    """Reshape the data to the specified new shape."""
    return data.reshape(new_shape)

def extract_unique_labels(dataframe, column_name):
    """Extract unique labels from a specified column in a DataFrame."""
    return dataframe[column_name].unique()

def average_across_sessions(data_list):
    """Average data across multiple sessions."""
    return np.mean(data_list, axis=0)

def load_and_process_data(file_paths):
    """Load data from given file paths and process it."""
    data_sessions = [pd.read_csv(fp) for fp in file_paths]
    return average_across_sessions(data_sessions)