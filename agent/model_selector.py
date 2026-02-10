from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def train_and_evaluate(X, y, problem_type):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if problem_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        }
    else:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
        }

    model_scores = {}
    trained_models = {}

    # Train & evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if problem_type == "Regression":
            score = r2_score(y_test, predictions)
        else:
            score = accuracy_score(y_test, predictions)

        model_scores[name] = score
        trained_models[name] = model

    # Select best
    best_model_name = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model_name]
    best_model = trained_models[best_model_name]

    # Feature importance
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(
            zip(X.columns, best_model.feature_importances_)
        )

    # ✅ Proper metrics dictionary (Cloud Safe)
    metrics = {
        "Best Model": best_model_name,
        "Best Score": round(float(best_score), 4),
        "All Model Scores": model_scores
    }

    # ✅ RETURN EXACTLY 3 VALUES
    return best_model, metrics, feature_importance
