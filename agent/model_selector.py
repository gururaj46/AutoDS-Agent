from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import pandas as pd
import numpy as np


def train_and_evaluate(X, y, problem_type, fast_mode=True):

    if y.nunique() <= 1:
        raise ValueError("Target column must contain more than 1 unique value.")

    # ==========================
    # 🔥 AUTO FEATURE REDUCTION
    # ==========================
    if X.shape[1] > 200:
        k = min(100, X.shape[1])  # keep only top 100 features
        if problem_type == "Regression":
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_classif, k=k)

        X = pd.DataFrame(selector.fit_transform(X, y))

    # ==========================
    # Train-Test Split
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==========================
    # ⚡ FAST MODE MODELS
    # ==========================
    if fast_mode:

        if problem_type == "Regression":
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
            }
            scoring = "r2"

        else:
            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Random Forest": RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42
                )
            }
            scoring = "accuracy"

        cv_folds = 3  # reduce CV

    else:
        # Slow full mode
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from xgboost import XGBClassifier, XGBRegressor

        if problem_type == "Regression":
            models = {
                "Random Forest": RandomForestRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, verbosity=0),
            }
            scoring = "r2"
        else:
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "XGBoost": XGBClassifier(random_state=42, verbosity=0, use_label_encoder=False),
            }
            scoring = "accuracy"

        cv_folds = 5

    # ==========================
    # Cross Validation
    # ==========================
    model_results = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
            model_results[name] = round(float(np.mean(scores)), 4)
        except:
            continue

    leaderboard = (
        pd.DataFrame(model_results.items(), columns=["Model", "Score"])
        .sort_values(by="Score", ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = models[best_model_name]

    # ==========================
    # Final Fit
    # ==========================
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    if problem_type == "Regression":
        best_score = round(float(r2_score(y_test, predictions)), 4)
    else:
        best_score = round(float(accuracy_score(y_test, predictions)), 4)

    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(
            sorted(
                zip(X.columns, best_model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
        )

    return best_model_name, best_score, leaderboard, feature_importance, best_model
