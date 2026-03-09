"""
Gradient Boosting Classification Demo
------------------------------------

A clean, professional example of Gradient Boosting for classification
using decision trees trained sequentially on residual errors
(negative gradients of the loss function).

Key Concepts:
- Residual (gradient) learning
- Additive ensemble of weak learners
- Learning rate (shrinkage)
- Bias reduction
- Probabilistic outputs

Author: Anuj Patel
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split


def generate_data():
    X, y = make_classification(
        n_samples=800,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    return X, y

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(
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
    probs = model.predict_proba(X_test)[:, 1]

    print("\nEvaluation Metrics")
    print("-" * 30)
    print(f"Accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall   : {recall_score(y_test, preds):.3f}")
    print(f"F1 Score : {f1_score(y_test, preds):.3f}")
    print(f"ROC AUC  : {roc_auc_score(y_test, probs):.3f}")

    print("\nClassification Report")
    print("-" * 30)
    print(classification_report(y_test, preds))

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.7)
    plt.title("Gradient Boosting Classification Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def main():
    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = train_gradient_boosting(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_decision_boundary(model, X_test, y_test)


if __name__ == "__main__":
    main()
