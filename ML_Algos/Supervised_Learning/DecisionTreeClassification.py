import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target


df['Target_Class'] = pd.qcut(df['Target'], q=3, labels=['Low', 'Medium', 'High'])
X = df[['MedInc']]                    
y = df['Target_Class']     

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nDecision Tree Classification Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)
y_range_pred = clf.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], df['Target_Class'].cat.codes, s=10, alpha=0.4, label="Data")

plt.plot(X_range, pd.Series(y_range_pred).map({'Low':0,'Medium':1,'High':2}),linewidth=3, color='red', label="Decision Tree Classification")

plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])
plt.xlabel("MedInc")
plt.ylabel("Target Class")
plt.title("Decision Tree Classification")
plt.legend()
plt.show()