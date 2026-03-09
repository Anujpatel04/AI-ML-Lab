import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def main():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['Target'] = data.target

    df['Class'] = pd.qcut(df['Target'], q=3, labels=[0, 1, 2])

    X = df[['MedInc']]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_clf = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        decision_function_shape='ovr',
        random_state=42
    )

    svm_clf.fit(X_train_scaled, y_train)

    y_test_pred = svm_clf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_test_pred)
    print("SVM Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, y_test_pred),
        annot=True,
        fmt='d',
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.show()

    X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)
    X_range_scaled = scaler.transform(X_range)
    y_range_pred = svm_clf.predict(X_range_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X['MedInc'],
        df['Class'],
        s=10,
        alpha=0.4,
        label="Data"
    )
    plt.plot(
        X_range,
        y_range_pred,
        color="red",
        linewidth=2,
        label="SVM Classification Boundary"
    )
    plt.xlabel("MedInc")
    plt.ylabel("Class (0=Low, 1=Mid, 2=High)")
    plt.title("SVM Classification (RBF Kernel)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
