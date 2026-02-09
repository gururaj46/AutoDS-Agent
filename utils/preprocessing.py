import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column):
    """
    Handles missing values and encodes categorical variables.
    Returns cleaned dataframe.
    """

    # Fill missing numeric values with mean
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y
