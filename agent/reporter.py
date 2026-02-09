def generate_report(summary, problem_type, metrics, feature_importance):
    """
    Generates structured business-style report.
    """

    report = "\n===== AUTO DATA SCIENTIST REPORT =====\n"

    report += f"\nDataset contains {summary['shape'][0]} rows and {summary['shape'][1]} columns."
    report += f"\nDetected problem type: {problem_type}."

    report += "\n\nModel Comparison Results:"
    for model_name, score in metrics["All Model Scores"].items():
        report += f"\n- {model_name}: {score}"

    report += f"\n\nBest Model Selected: {metrics['Best Model']}"
    report += f"\nBest Score: {metrics['Best Score']}"

    if feature_importance:
        report += "\n\nFeature Importance (Top Drivers):"
        for feature, importance in feature_importance.items():
            report += f"\n- {feature}: {importance}"

    report += "\n\nMissing Values Detected:"
    for col, val in summary['missing_values'].items():
        if val > 0:
            report += f"\n- {col}: {val} missing values"

    report += "\n\nRecommendation:"
    report += "\n- Consider hyperparameter tuning and advanced feature engineering."

    report += "\n\n======================================"

    return report
