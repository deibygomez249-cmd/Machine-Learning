# ML Explorer - Machine Learning Web Application

Interactive Flask web application demonstrating supervised and unsupervised Machine Learning models with Bootstrap 5 and scikit-learn.

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
- Bootstrap 5 (CDN only)

## Project Structure

R1A1/
├── app.py
├── WeightRegression.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── generate_datasets.py
│   ├── process_used_cars.py
│   ├── linear_regression.csv
│   ├── logistic_regression.csv
│   ├── ridge_classifier.csv
│   └── used_cars_clean.csv
├── models/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── ridge_classifier.py
│   └── clustering.py
└── templates/
    ├── base.html
    ├── index.html
    ├── comparison.html
    ├── machine_learning/
    ├── use_cases/
    ├── linear_regression/
    ├── logistic_regression/
    ├── ridge_classifier/
    └── unsupervised/

## Installation

git clone <repository-url>
cd R1A1
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Datasets

### Supervised Models

| File | Records | Independent Variable | Target Variable |
|------|---------|----------------------|-----------------|
| linear_regression.csv | 500 | shipping_cost | package_weight |
| logistic_regression.csv | 500 | study_hours | passed_exam |
| ridge_classifier.csv | 500 | study_hours, attendance, previous_grade, assignments_completed | academic_risk |

### Unsupervised Model

| File | Records | Clustering Variables | Source |
|------|---------|----------------------|--------|
| used_cars_clean.csv | 4009 | vehicle_age, mileage | Used Car Price Prediction Dataset (Kaggle) |

To regenerate the supervised datasets:

python data/generate_datasets.py

To process the used-car dataset (download first from Kaggle):

python data/process_used_cars.py

## Machine Learning Models

### Supervised

- Weight Regression (Activity 1) - predicts package weight from shipping cost
- Linear Regression - predicts package weight from shipping cost
- Logistic Regression - binary classification of student exam result
- Ridge Classifier - academic risk classification

### Unsupervised

- K-Means Clustering - used-car market segmentation

All models use an 80/20 train-test split (random_state=42). Predictions and metrics are computed dynamically.

## Unsupervised Learning Module

The K-Means module segments used vehicles based on Vehicle Age (years) and Mileage (miles).

Components:

- Concepts: theoretical background of K-Means
- Manual Exercise: step-by-step manual K-Means on 100 records across 3 iterations
- Clustering Application: full dataset clustering with scikit-learn

Reference year for Vehicle Age: 2026

Vehicle Age = 2026 - model_year

## Running Locally

python app.py

Open http://localhost:5000

## Git Workflow

git checkout main
git pull origin main
git checkout -b feature/unsupervised-machine-learning

git add .
git commit -m "Add used car dataset preprocessing"
git commit -m "Implement K-Means clustering module"
git commit -m "Add unsupervised learning templates"
git commit -m "Integrate unsupervised module into navigation"

git push -u origin feature/unsupervised-machine-learning

Open a Pull Request from feature/unsupervised-machine-learning into main on GitHub and merge it.

## Render Deployment

- Build Command: pip install -r requirements.txt
- Start Command: gunicorn app:app

Ensure the data/ folder with all CSV files is committed to the repository.