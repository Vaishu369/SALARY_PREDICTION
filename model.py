import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\rahitya\OneDrive\Desktop\ML_PRO\salary_prediction_updated_dataset.csv")

print("Initial Data Preview:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

df['YearsExperience'].fillna(df['YearsExperience'].median(), inplace=True)
df['Job Role'].fillna(df['Job Role'].mode()[0], inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)

df.drop_duplicates(inplace=True)

Q1 = df['YearsExperience'].quantile(0.25)
Q3 = df['YearsExperience'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['YearsExperience'] >= lower_bound) & (df['YearsExperience'] <= upper_bound)]

X = df[['YearsExperience', 'Job Role']]
y = df['Salary']

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = encoder.fit_transform(X[['Job Role']])

X_final = np.hstack((X[['YearsExperience']].values, X_encoded))

scaler = StandardScaler()
X_final = scaler.fit_transform(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

joblib.dump(model, "salary_model.pkl")
joblib.dump(encoder, "encoder.pkl") 
joblib.dump(scaler, "scaler.pkl")
print("\nModel, encoder, and scaler saved successfully!")

experience = np.array([[7]])
job_role = np.array(["Data Scientist"]).reshape(-1, 1)

job_role_encoded = encoder.transform(job_role)
input_data = np.hstack((experience, job_role_encoded))
input_data = scaler.transform(input_data)

predicted_salary = model.predict(input_data)
print(f"\nPredicted Salary for {experience[0][0]} years as {job_role[0][0]}: ${predicted_salary[0]:,.2f}")

plt.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Actual Data')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary Prediction based on Experience & Job Role")
plt.legend()
plt.show()