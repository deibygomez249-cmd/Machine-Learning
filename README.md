# ML Explorer — Activities 1 & 2

Interactive Flask web application that demonstrates Machine Learning concepts with four supervised models: Weight Regression (Activity 1), Linear Regression, Logistic Regression and Ridge Classifier (Activity 2).

## Authors

- Andres Julian Forero Gacha
- Deiby Esteban Gomez Naranjo

## Technologies

- Python 3.11
- Flask
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Bootstrap 5

## Installation

git clone <repository-url>
cd R1A2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/generate_datasets.py

## Running Locally

python app.py

Then open http://localhost:5000

## Dataset Structure

All Activity 2 datasets live in data/ and contain 500 records each.

| File | Records | Independent Variable | Target Variable |
|------|---------|----------------------|-----------------|
| linear_regression.csv | 500 | shipping_cost | package_weight |
| logistic_regression.csv | 500 | study_hours | passed_exam |
| ridge_classifier.csv | 500 | study_hours, attendance, previous_grade, assignments_completed | academic_risk |

## Machine Learning Models

- Weight Regression (Activity 1) — predicts package weight from shipping cost.
- Linear Regression (Activity 2) — predicts package weight from shipping cost using a CSV dataset.
- Logistic Regression — predicts if a student passes based on study hours.
- Ridge Classifier — predicts academic risk from four features.

All Activity 2 models use an 80/20 train-test split (random_state=42). Predictions and metrics are computed dynamically at runtime.

## Git Workflow

master → R1A2 → Development → Commits → Push → Pull Request → Merge → master

## Deployment (Render)

- Build Command: pip install -r requirements.txt
- Start Command: gunicorn app:app
- Ensure data/*.csv files are committed to the repository.
# commit trigger