def detect_problem_type(df, target_column):
    """
    Detects whether problem is Classification or Regression
    based on target column data type.
    """

    if df[target_column].dtype == "object":
        return "Classification"
    else:
        return "Regression"
