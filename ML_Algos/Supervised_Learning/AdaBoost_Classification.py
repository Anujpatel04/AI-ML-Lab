"""
AdaBoost Classification 
----------------------------

This script demonstrates a clean, professional implementation of the
AdaBoost boosting algorithm using decision stumps as weak learners.

Key Concepts Demonstrated:
- Weak learners (decision stumps)
- Sequential learning
- Error-focused reweighting
- Ensemble prediction

Author: Anuj Patel
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def generate_data():
    """
    Generate a simple 2D classification dataset.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42
    )
    return X, y


def train_adaboost(X_train, y_train):
    """
    Train AdaBoost with decision stumps as weak learners.
    """
    base_learner = DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    )

    model = AdaBoostClassifier(
        estimator=base_learner,
        n_estimators=50,
        learning_rate=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance.
    """
    predictions = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

def plot_decision_boundary(model, X, y):
    """
    Plot decision boundary for 2D data.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.title("AdaBoost Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def main():
    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = train_adaboost(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    plot_decision_boundary(model, X_test, y_test)


if __name__ == "__main__":
    main()
