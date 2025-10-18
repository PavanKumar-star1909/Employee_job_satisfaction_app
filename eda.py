import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define project root for absolute paths (update if your structure differs)
project_root = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction'

# Load cleaned data (using absolute path)
data_path = os.path.join(project_root, 'data', 'cleaned_employee_data.csv')
df = pd.read_csv(data_path)

print("EDA on Cleaned Dataset")
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Create plots folder if it doesn't exist
plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# 1. Univariate Analysis: Histograms for key numeric features
numeric_cols = ['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction']
df[numeric_cols].hist(figsize=(12, 8))
plt.suptitle('Histograms of Key Numeric Features')
plt.savefig(os.path.join(plots_dir, 'histograms.png'))  # Save with absolute path
plt.show()

# 2. Categorical Analysis: Attrition distribution
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution (0=Stay, 1=Leave)')
plt.savefig(os.path.join(plots_dir, 'attrition_count.png'))
plt.show()

# 3. Bivariate Analysis: Correlation heatmap
plt.figure(figsize=(20, 15))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm')  # annot=False for speed; set to True for values
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
plt.show()

# 4. Key Insights: Attrition by Department (using original raw data for visualization, since we encoded)
# Load raw data for categorical plots (using absolute path)
raw_path = os.path.join(project_root, 'data', 'Employee_Attrition.csv')
raw_df = pd.read_csv(raw_path)
attrition_rate = raw_df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean()).reset_index()
sns.barplot(x='Department', y='Attrition', data=attrition_rate)
plt.title('Attrition Rate by Department')
plt.savefig(os.path.join(plots_dir, 'attrition_by_dept.png'))
plt.show()

# 5. Scatter Plot: Age vs. MonthlyIncome, colored by Attrition
sns.scatterplot(x='Age', y='MonthlyIncome', hue='Attrition', data=df)
plt.title('Age vs. Monthly Income by Attrition')
plt.savefig(os.path.join(plots_dir, 'age_income_scatter.png'))
plt.show()

print("EDA Complete. Key Insights: Check correlations (e.g., JobSatisfaction negatively correlates with Attrition). Plots saved in 'plots/' folder.")
