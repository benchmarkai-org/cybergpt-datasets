import pandas as pd
import numpy as np
    

def feature_df_to_numpy(features_df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to numpy array with one-hot encoding for categorical variables."""
    df = features_df.copy()
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    else:
        encoded = df
        
    return encoded.values