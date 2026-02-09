from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np


def train_and_evaluate(X, y, problem_type):
    """
    Trains multiple models, compares performance,
    selects best model automatically.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    if problem_type == "Classification":
        models["RandomForest"] = RandomForestClassifier(random_state=42)
        models["LogisticRegression"] = LogisticRegression(max_iter=1000)

    else:
        models["RandomForest"] = RandomForestRegressor(random_state=42)
        models["LinearRegression"] = LinearRegression()

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if problem_type == "Classification":
            score = accuracy_score(y_test, predictions)
        else:
            score = r2_score(y_test, predictions)

        results[name] = round(float(score), 4)
        trained_models[name] = model

    # Select best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    best_score = results[best_model_name]

    metrics = {
        "Best Model": best_model_name,
        "Best Score": best_score,
        "All Model Scores": results
    }

    # Feature Importance (only if available)
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        importance_values = best_model.feature_importances_
        for feature, importance in zip(X.columns, importance_values):
            feature_importance[feature] = round(float(importance), 4)

        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    return best_model, metrics, feature_importance
