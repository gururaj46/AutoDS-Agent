def detect_problem_type(df, target_column):
    """
    Detect whether problem is Classification or Regression
    using smarter logic based on data type + uniqueness.
    """

    y = df[target_column]

    # -------------------------
    # If numeric column
    # -------------------------
    if y.dtype in ["int64", "float64"]:

        unique_values = y.nunique()

        # Small number of unique values → likely classification
        if unique_values <= 15:
            return "Classification"

        # Large number of unique numeric values → regression
        return "Regression"

    # -------------------------
    # If object / string column
    # -------------------------
    else:

        unique_ratio = y.nunique() / len(y)

        # If too many unique values → probably ID column
        if unique_ratio > 0.5:
            raise ValueError(
                "❌ Selected target column looks like an ID/text column.\n"
                "Please select a proper Classification or Regression target."
            )

        return "Classification"
