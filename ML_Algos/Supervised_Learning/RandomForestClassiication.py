import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

df['Class'] = pd.qcut(df['Target'], q=3, labels=[0, 1, 2])

X = df[['MedInc']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
rf_clf.fit(X_train, y_train)

y_test_pred = rf_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
print("Random Forest Classification Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)
y_range_pred = rf_clf.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], df['Class'], s=10, alpha=0.4, label="Data")
plt.plot(X_range, y_range_pred, color="red", linewidth=2, label="RF Classification Boundary")
plt.xlabel("MedInc")
plt.ylabel("Class (0=Low, 1=Mid, 2=High)")
plt.title("Random Forest Classification")
plt.legend()
plt.show()
