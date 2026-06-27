import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(r"C:\Users\rahitya\OneDrive\Desktop\salary prediction\salary_prediction_updated_dataset.csv")
print(df.head())

X = df[['YearsExperience', 'Job Role']]
y = df['Salary']

# Corrected the argument to sparse_output=False
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X[['Job Role']])

X_final = np.hstack((X[['YearsExperience']].values, X_encoded))

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

experience = np.array([[7]])
job_role = np.array([["Data Scientist"]])

job_role_encoded = encoder.transform(job_role)
input_data = np.hstack((experience, job_role_encoded))

predicted_salary = model.predict(input_data)
print(f"Predicted Salary for {experience[0][0]} years as {job_role[0][0]}: {predicted_salary[0]:,.2f}")

plt.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Actual Data')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction based on Experience & Job Role")
plt.legend()
plt.show()