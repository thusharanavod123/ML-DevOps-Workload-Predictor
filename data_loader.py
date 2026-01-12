import pandas as pd
import os

def load_data(filepath):
    """
    Loads the cloud workload dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Ensure the date column is parsed if it exists (optional but good practice)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    return df