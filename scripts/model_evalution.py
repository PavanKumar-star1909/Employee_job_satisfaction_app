import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
import joblib
from imblearn.over_sampling import SMOTE

# Paths
data_path = r"C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\data\engineered_employee_data.csv"
model_path = r"C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\models\attrition_model.pkl"
metrics_dir = r"C:\Users\vishv\Downloads\python_learning.1st project\__MACOSX\Employee Attrition Analysis and Prediction\docs"
metrics_file = os.path.join(metrics_dir, "model_metrics.txt")

os.makedirs(metrics_dir, exist_ok=True)

print("Starting Model Evaluation...")

# Load data and model
df = pd.read_csv(data_path)
model = joblib.load(model_path)

# Prepare data
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Handle class imbalance for evaluation (optional)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Predict
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc_roc:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

# Example Predictions
print("\nExample Predictions:")
sample = X_test.iloc[0:3]
sample_pred = model.predict(sample)
sample_proba = model.predict_proba(sample)[:, 1]

for i, (pred, proba) in enumerate(zip(sample_pred, sample_proba)):
    print(f"Sample {i+1}: Predicted Attrition={pred}, Probability={proba:.2f}")

# Save metrics
with open(metrics_file, "w") as f:
    f.write("Model Evaluation Metrics\n")
    f.write("=========================\n")
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write(f"Precision: {precision:.2f}\n")
    f.write(f"Recall: {recall:.2f}\n")
    f.write(f"F1-Score: {f1:.2f}\n")
    f.write(f"AUC-ROC: {auc_roc:.2f}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

print(f"\n✅ Metrics successfully saved to:\n{metrics_file}")
