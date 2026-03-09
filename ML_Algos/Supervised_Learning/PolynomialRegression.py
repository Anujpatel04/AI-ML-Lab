import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
print(df.head())

X = df[['MedInc']]
y = df['Target']

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
from sklearn.model_selection import train_test_split

X_poly_train, X_poly_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

poly_model.fit(X_poly_train, y_train)

X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)
X_range_poly = poly_features.transform(X_range)
y_pred = poly_model.predict(X_range_poly)

y_test_pred = poly_model.predict(X_poly_test)

mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Model Evaluation Metrics on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

sorted_idx = np.argsort(X_range.flatten())
X_range_sorted = X_range.flatten()[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], y, s=10, alpha=0.4, label='Data')
plt.plot(X_range_sorted, y_pred_sorted, color='red', linewidth=3, label='Polynomial Regression')
plt.legend()
plt.show()