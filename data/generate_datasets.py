import os
import numpy as np
import pandas as pd

np.random.seed(42)
os.makedirs('data', exist_ok=True)
n = 500

shipping_cost = np.random.uniform(5000, 50000, n).round(2)
package_weight = (shipping_cost * 0.0002 + np.random.normal(0, 0.3, n)).round(3)
package_weight = np.clip(package_weight, 0.5, 15)
pd.DataFrame({
    'shipping_cost': shipping_cost,
    'package_weight': package_weight
}).to_csv('data/linear_regression.csv', index=False)

study_hours_lr = np.random.uniform(1, 15, n).round(2)
logit = 0.8 * study_hours_lr - 5
prob_pass = 1 / (1 + np.exp(-logit))
passed_exam = (np.random.uniform(0, 1, n) < prob_pass).astype(int)
pd.DataFrame({
    'study_hours': study_hours_lr,
    'passed_exam': passed_exam
}).to_csv('data/logistic_regression.csv', index=False)

study_hours_r = np.random.uniform(1, 20, n).round(2)
attendance = np.random.uniform(40, 100, n).round(2)
previous_grade = np.random.uniform(2.0, 5.0, n).round(2)
assignments_completed = np.random.randint(0, 11, n)
score = (-0.15 * study_hours_r - 0.04 * attendance - 0.5 * previous_grade - 0.3 * assignments_completed + 5.5)
prob_risk = 1 / (1 + np.exp(-score))
academic_risk = (np.random.uniform(0, 1, n) < prob_risk).astype(int)
pd.DataFrame({
    'study_hours': study_hours_r,
    'attendance': attendance,
    'previous_grade': previous_grade,
    'assignments_completed': assignments_completed,
    'academic_risk': academic_risk
}).to_csv('data/ridge_classifier.csv', index=False)

print('Datasets generated successfully.')