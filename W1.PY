import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load data
insurance = pd.read_csv('insurance.csv.xls')

# One-hot encoding
catColumns = ['sex', 'smoker', 'region']
insurance_dum = pd.get_dummies(insurance, columns=catColumns, drop_first=True)

# Features and target
X = insurance_dum[['age', 'bmi', 'smoker_yes']]
y = insurance_dum['charges']

# ----------- TRAIN TEST SPLIT -----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predictions
y_train_pred = linreg.predict(X_train)
y_test_pred = linreg.predict(X_test)

# ----------- METRICS (TEST DATA) -----------
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1-r2) * (n-1)/(n-p-1)

print("TEST R2 Score:", r2)
print("Adjusted R2:", adj_r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# ----------- COEFFICIENTS -----------
print("Coefficients:", linreg.coef_)
print("Intercept:", linreg.intercept_)

# ----------- PLOTS (TEST DATA) -----------

# 1. Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_test_pred)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted (Test Data)")
plt.show()

# 2. Residuals vs Predicted
residuals = y_test - y_test_pred
plt.figure()
plt.scatter(y_test_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (Test Data)")
plt.show()

# 3. Residual Distribution
plt.figure()
plt.hist(residuals, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Distribution (Test Data)")
plt.show()

# 4. Coefficient Bar Plot
plt.figure()
plt.bar(X.columns, linreg.coef_)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Impact")
plt.show()

# ----------- PREDICTION FUNCTION -----------

def calc_insurance(age, bmi, smoking):
    y = (age * linreg.coef_[0] +
         bmi * linreg.coef_[1] +
         smoking * linreg.coef_[2] +
         linreg.intercept_)
    return y

print("Predicted Charge:", calc_insurance(20, 24, 0))
