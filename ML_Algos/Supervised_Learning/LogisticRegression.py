import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
print(df.head())

median_price = df['Target'].median()
df['PriceClass'] = (df['Target'] >= median_price).astype(int)

X = df[['MedInc']]
y = df['PriceClass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)

y_prob = log_reg.predict_proba(X_range)[:, 1]

plt.figure(figsize=(10,6))
plt.scatter(X['MedInc'], y, alpha=0.3, label="Data (0/1)")
plt.plot(X_range, y_prob, color='red', linewidth=3, label="Logistic Regression Curve")
plt.xlabel("Median Income")
plt.ylabel("Probability of High Price")
plt.legend()
plt.show()
