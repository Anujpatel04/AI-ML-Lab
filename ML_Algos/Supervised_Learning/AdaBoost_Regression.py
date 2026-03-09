"""
AdaBoost Regression Demo
-----------------------

A professional, minimal implementation of AdaBoost for regression using
decision tree regressors as weak learners.

Concepts Demonstrated:
- Boosting for regression
- Weak learners (shallow trees)
- Sequential error correction
- Bias reduction via ensembles

Author: Anuj Patel
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def generate_data():
    """
    Generate a noisy regression dataset.
    """
    X, y = make_regression(
        n_samples=600,
        n_features=1,
        n_informative=1,
        noise=20.0,
        random_state=42
    )
    return X, y

def train_adaboost_regressor(X_train, y_train):
    """
    Train AdaBoost Regressor with weak learners.
    """
    base_learner = DecisionTreeRegressor(
        max_depth=2,  # weak learner
        random_state=42
    )

    model = AdaBoostRegressor(
        estimator=base_learner,
        n_estimators=100,
        learning_rate=0.8,
        loss="linear",  # linear | square | exponential
        random_state=42
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate regression performance.
    """
    preds = model.predict(X_test)

    print("\nEvaluation Metrics")
    print("-" * 30)
    print(f"MSE  : {mean_squared_error(y_test, preds):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test, preds):.3f}")
    print(f"R²   : {r2_score(y_test, preds):.3f}")

def plot_regression(model, X, y):
    """
    Plot regression curve vs actual data.
    """
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_pred = model.predict(X_plot)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.4, label="Actual Data")
    plt.plot(X_plot, y_pred, color="red", linewidth=2, label="AdaBoost Prediction")
    plt.title("AdaBoost Regression")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.show()

def main():
    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = train_adaboost_regressor(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_regression(model, X_test, y_test)


if __name__ == "__main__":
    main()
