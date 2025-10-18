import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load cleaned data
data_path = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\data\cleaned_employee_data.csv'
df = pd.read_csv(data_path)

print("Starting Feature Engineering")
print(f"Original Shape: {df.shape}")

# 1. Create Tenure Categories (based on YearsAtCompany)
df['TenureCategory'] = pd.cut(df['YearsAtCompany'], bins=[-1, 2, 5, 10, 20, 50], labels=['New', 'Mid', 'Experienced', 'Senior', 'Veteran'])

# 2. Create Engagement Score (average of JobInvolvement and WorkLifeBalance)
df['EngagementScore'] = (df['JobInvolvement'] + df['WorkLifeBalance']) / 2

# 3. Encode new categoricals (one-hot for TenureCategory)
df = pd.get_dummies(df, columns=['TenureCategory'], drop_first=True)

# 4. Optional: Feature Selection (drop low-correlation features if needed)
corr_with_attrition = df.corr()['Attrition'].abs().sort_values(ascending=False)
low_corr_features = corr_with_attrition[corr_with_attrition < 0.1].index.tolist()
df.drop(low_corr_features, axis=1, inplace=True)
print(f"Dropped low-correlation features: {low_corr_features}")

# 5. Re-scale if new numeric features added (EngagementScore)
scaler = StandardScaler()
df[['EngagementScore']] = scaler.fit_transform(df[['EngagementScore']])

print(f"Final Shape after Feature Engineering: {df.shape}")
print(df.head())

# Save updated data
engineered_path = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\data\engineered_employee_data.csv'
df.to_csv(engineered_path, index=False)
print(f"Engineered data saved to: {engineered_path}")
