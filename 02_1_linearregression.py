import numpy as np
import pandas as pd

data = pd.read_csv('Salary_Data.csv').to_numpy()

row_count = data.shape[0]

X = data[:, :-1] 
intercept_column = np.ones((row_count, 1))
X = np.hstack([X, intercept_column])

y = data[:, -1]
y = y.reshape(row_count, 1)

# I am using an 80% train, 20% test dataset split:
split_index = int(0.8 * row_count)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#'Least Squares' method…
XᵀX = X_train.T @ X_train
Xᵀy = X_train.T @ y_train
weights = np.linalg.inv(XᵀX) @ Xᵀy

y_predicted = X_test @ weights

for i in range(len(y_test)):
    error = y_predicted[i][0] - y_test[i][0]
    percentage_error = 100 * error / y_test[i][0]
    print(f"{y_test[i][0]:.0f} -> {y_predicted[i][0]:.2f}\tError: {error:.2f} ({percentage_error:.1f}%)")

coefficients = weights[:-1, 0]
print(f"\nCoefficients (m1, m2, ...): {coefficients}")
intercept = weights[-1, 0]           
print(f"Intercept (c): {intercept}")

rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
print(f"\nRoot Mean Squared Error on Test Set: {rmse}\n")
