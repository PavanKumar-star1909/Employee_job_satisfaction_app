
# Employee Attrition Analysis and Prediction 📊

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview
Employee turnover poses a significant challenge for organizations, resulting in increased costs, reduced productivity, and team disruptions. This project analyzes employee data, identifies key drivers of attrition, and builds predictive models to enable HR teams to proactively retain employees and optimize workforce planning.

---

## Skills Gained
- **Data Preprocessing and Cleaning**: Handling missing values, outliers, and categorical encoding.
- **Exploratory Data Analysis (EDA)**: Visualizing patterns, correlations, and distributions.
- **Feature Engineering**: Creating new features like tenure categories and engagement scores.
- **Machine Learning Model Development**: Implementing and tuning models (e.g., Random Forest, Logistic Regression).
- **Model Evaluation**: Assessing performance with metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
- **Streamlit Application Development**: Building interactive dashboards for HR insights.

---

## Domain
**HR Analytics**: Leveraging data science to predict and prevent employee attrition, supporting data-driven HR strategies.

---

## Problem Statement
High employee turnover leads to elevated recruitment, onboarding, and training costs while disrupting productivity. This project focuses on:
- Analyzing factors influencing employee attrition.
- Predicting at-risk employees using machine learning.
- Providing actionable insights for retention strategies.

---

## Business Use Cases
- **Employee Retention**: Identify high-risk employees and deploy targeted interventions (e.g., bonuses, career development).
- **Cost Optimization**: Minimize expenses from hiring and training by reducing turnover.
- **Workforce Planning**: Use predictive insights to align HR strategies with business goals and improve employee satisfaction.

---

## Approach

1. **Data Collection & Preprocessing**:
   - Gathered employee data including demographics, job roles, performance, and tenure.
   - Cleaned data: Handled missing values, removed outliers, encoded categoricals, and scaled features.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed distributions, correlations, and attrition patterns (e.g., via histograms, heatmaps, and scatter plots).

3. **Feature Engineering**:
   - Created features like `TenureCategory` and `EngagementScore` to enhance model performance.

4. **Model Building**:
   - Trained models (e.g., Random Forest) with hyperparameter tuning using GridSearchCV.

5. **Model Evaluation**:
   - Evaluated using accuracy, precision, recall, F1-score, AUC-ROC, and confusion matrix.

6. **Deployment**:
   - Developed an interactive Streamlit dashboard for predictions and visualizations.

---

## Dataset
- **Source**: [IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (or your local file: `data/Employee_Attrition.csv`).
- **Description**: Contains 1,470 employee records with 35 features related to demographics, job attributes, and performance.
- **Key Features**:
  - `Age`, `Gender`, `Department`, `JobRole`, `JobSatisfaction`, `OverTime`, `YearsAtCompany`, `MonthlyIncome`, `PerformanceRating`, `WorkLifeBalance`, etc.
- **Target Variable**: `Attrition` (0 = Stayed, 1 = Left).

---

## Results
- **Predictive Model Accuracy**: 84% on test data (Random Forest with tuned hyperparameters).
- **Key Metrics**:
  - Precision: 73%
  - Recall: 17%
  - F1-Score: 27%
  - AUC-ROC: 0.81
  - Confusion Matrix: [[221, 3], [40, 8]]
- **Key Drivers of Attrition**: Low job satisfaction, frequent overtime, poor work-life balance, and inadequate compensation.
- **Interactive Dashboards**: Streamlit app provides visualizations (e.g., attrition trends, at-risk employees) and predictions for HR decision-making.

---

## Folder Structure
```
Employee Attrition Analysis and Prediction/
│
├── data/                          # Raw and processed datasets (e.g., cleaned_employee_data.csv, engineered_employee_data.csv)
├── models/                        # Saved ML models (e.g., attrition_model.pkl) and feature lists
├── plots/                         # Visualizations (e.g., histograms.png, correlation_heatmap.png, feature_importance.png)
├── scripts/                       # Python scripts for each phase (e.g., preprocessing.py, eda.py, feature_engineering.py, model_building.py, model_evaluation.py)
├── src/                           # Streamlit app code (e.g., app.py)
├── docs/                          # Documentation and metrics (e.g., model_metrics.txt, README.md)
├── requirements.txt               # Python dependencies
└── .gitignore                     # Files to ignore in Git (e.g., data/, models/)
```

---

## How to Run

### Prerequisites
- Python 3.8+ installed.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/employee-attrition-analysis.git
   cd employee-attrition-analysis
   ```

2. **Run Scripts Sequentially** (for full workflow):
   - Preprocessing: `python scripts/preprocessing.py`
   - EDA: `python scripts/eda.py`
   - Feature Engineering: `python scripts/feature_engineering.py`
   - Model Building: `python scripts/model_building.py`
   - Model Evaluation: `python scripts/model_evaluation.py`

3. **Launch the Streamlit App**:
   ```bash
   streamlit run src/app.py
   ```
   - Open the provided URL in your browser to interact with the dashboard (e.g., input employee details for attrition predictions).

### Example Usage
- Input an employee's details (e.g., Age: 35, MonthlyIncome: 5000, JobSatisfaction: Low) to get a predicted attrition probability.
- View EDA plots and at-risk employee lists.

---

## Future Improvements
- Integrate advanced models like XGBoost or LightGBM for better accuracy.
- Add real-time data pipelines for continuous attrition monitoring.
- Develop a recommendation engine for personalized retention strategies (e.g., based on employee feedback).
- Expand to multi-class predictions (e.g., performance rating or promotion likelihood).

---

## Contributing
Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.


```
