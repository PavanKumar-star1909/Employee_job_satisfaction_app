import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt

project_root = r'C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction'

# Load engineered data
data_path = os.path.join(project_root, 'data', 'engineered_employee_data.csv')
df = pd.read_csv(data_path)

print("Starting Model Building for Attrition Prediction")

# Prepare data
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"After SMOTE, class distribution:\n{pd.Series(y_res).value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [10, 20, None],
              'min_samples_split': [2, 5]}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print(f"Best Parameters: {grid.best_params_}")
print(f"Best CV Score: {grid.best_score_}")

# Save model
models_dir = os.path.join(project_root, 'models')
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'attrition_model.pkl')
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Save feature columns for Streamlit app
features_path = os.path.join(models_dir, 'model_features.pkl')
joblib.dump(X.columns.tolist(), features_path)
print(f"Feature columns saved to: {features_path}")

# Feature importance plot
feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_importances.head(10).plot(kind='bar')
plt.title('Top 10 Feature Importances')

plots_dir = os.path.join(project_root, 'plots')
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
plt.show()
