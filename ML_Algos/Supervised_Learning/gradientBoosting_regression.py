"""
Gradient Boosting Regression Demo
--------------------------------

A clean, professional example of Gradient Boosting using decision trees
trained sequentially on residual errors.

Key Concepts:
- Residual learning
- Additive model building
- Learning rate (shrinkage)
- Bias reduction

Author: Anuj Patel
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# -----------------------------
# 1. Data Generation
# -----------------------------
def generate_data():
    X, y = make_regression(
        n_samples=700,
        n_features=1,
        noise=25,
        random_state=42
    )
    return X, y


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        subsample=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\nEvaluation Metrics")
    print("-" * 30)
    print(f"MSE  : {mean_squared_error(y_test, preds):.3f}")
    print(f"MAE  : {mean_absolute_error(y_test, preds):.3f}")
    print(f"R²   : {r2_score(y_test, preds):.3f}")

def plot_regression(model, X, y):
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_pred = model.predict(X_plot)

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.4, label="Actual Data")
    plt.plot(X_plot, y_pred, color="green", linewidth=2, label="Gradient Boosting")
    plt.title("Gradient Boosting Regression")
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

    model = train_gradient_boosting(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_regression(model, X_test, y_test)


if __name__ == "__main__":
    main()
