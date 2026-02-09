import pandas as pd

def analyze_data(file_path):
    """
    Loads dataset and performs basic EDA.
    Returns dataframe and summary dictionary.
    """

    df = pd.read_csv(file_path)

    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "basic_statistics": df.describe(include="all").to_dict()
    }

    return df, summary
