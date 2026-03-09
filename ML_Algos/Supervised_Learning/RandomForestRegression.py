import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

X = df[['MedInc']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
rf_model.fit(X_train, y_train)

y_test_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Random Forest Regression Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

X_range = np.linspace(X['MedInc'].min(), X['MedInc'].max(), 300).reshape(-1, 1)
y_range_pred = rf_model.predict(X_range)

plt.figure(figsize=(10, 6))
plt.scatter(X['MedInc'], y, s=10, alpha=0.4, label="Data")
plt.plot(X_range, y_range_pred, linewidth=3, color='red', label="Random Forest Regression")
plt.legend()
plt.xlabel("MedInc")
plt.ylabel("Target")
plt.title("Random Forest Regression")
plt.show()
