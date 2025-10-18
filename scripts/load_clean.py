import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the raw dataset
# Use the exact path you provided
file_path = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\data\Employee_Attrition.csv'
df = pd.read_csv(file_path)

# Print initial summaries (as in the original code)
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Step 2: Handle Missing Values
# Check for missing values (none expected in this dataset, but always good to verify)
print("\nMissing Values per Column:")
print(df.isnull().sum())

# If there were missing values, you could impute them like this (commented out as not needed):
# df['Age'].fillna(df['Age'].median(), inplace=True)  # Example for numeric
# df['Department'].fillna(df['Department'].mode()[0], inplace=True)  # Example for categorical

# Step 3: Handle Outliers
# Example: Remove outliers for MonthlyIncome using IQR method
# You can add similar logic for other numeric columns if needed (e.g., Age, YearsAtCompany)
print(f"\nOriginal dataset shape: {df.shape}")
Q1 = df['MonthlyIncome'].quantile(0.25)
Q3 = df['MonthlyIncome'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((df['MonthlyIncome'] < lower_bound) | (df['MonthlyIncome'] > upper_bound))]
print(f"Dataset shape after outlier removal: {df.shape}")

# Step 4: Encode Categorical Variables
# Encode binary categoricals (e.g., Attrition: 'Yes'=1, 'No'=0)
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # Yes=1, No=0

# One-hot encode non-ordinal categoricals (e.g., Department, EducationField, etc.)
# drop_first=True to avoid multicollinearity
df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus', 'Gender', 'OverTime'], drop_first=True)

# Ordinal columns like JobSatisfaction (1-4) are already numeric, so no change needed.

# Step 5: Drop Irrelevant Columns
# Drop constants or non-useful columns as per the dataset description
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
# Note: EmployeeNumber is unique per employee, so it's dropped to avoid overfitting in models.

# Step 6: Normalize/Scale Numeric Features
# Scale numeric columns for models sensitive to scale (e.g., Logistic Regression)
scaler = StandardScaler()
numeric_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 7: Final Checks and Save Cleaned Data
print("\nCleaned Dataset Info:")
print(df.info())
print("\nFirst 5 rows of cleaned dataset:")
print(df.head())

# Save the cleaned dataset in the SAME folder as the raw data
cleaned_file_path = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\data\cleaned_employee_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to: {cleaned_file_path}")

# Optional: Print shape and confirm no missing values post-cleaning
print(f"\nFinal dataset shape: {df.shape}")
print("Missing values after cleaning:")
print(df.isnull().sum())
