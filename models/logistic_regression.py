import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

DATA_PATH = os.path.join('data', 'logistic_regression.csv')


def load_logistic_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError('Logistic regression dataset not found. Run data/generate_datasets.py')
    df = pd.read_csv(DATA_PATH).dropna()
    return df


def train_logistic_model():
    df = load_logistic_data()
    X = df[['study_hours']].values
    y = df['passed_exam'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    info = {
        'records': int(len(df)),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'class_0_meaning': 'Did Not Pass',
        'class_1_meaning': 'Passed',
        'independent': 'Study Hours',
        'target': 'Passed Exam'
    }
    return model, info


def predict_exam_result(model, study_hours):
    value = float(study_hours)
    if value < 0:
        raise ValueError('Study hours cannot be negative.')
    prediction = int(model.predict([[value]])[0])
    probability = float(model.predict_proba([[value]])[0][prediction])
    return prediction, round(probability * 100, 2)


def get_logistic_metrics(model):
    df = load_logistic_data()
    X = df[['study_hours']].values
    y = df['passed_exam'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'confusion_matrix': cm.tolist(),
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    }
    return metrics


def generate_logistic_plot(model):
    df = load_logistic_data()
    X = df['study_hours'].values
    y = df['passed_exam'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    class0 = X[y == 0]
    class1 = X[y == 1]
    ax.scatter(class0, np.zeros_like(class0), color='#dc3545', alpha=0.4, s=30, label='Did Not Pass (0)')
    ax.scatter(class1, np.ones_like(class1), color='#198754', alpha=0.4, s=30, label='Passed (1)')

    x_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    proba = model.predict_proba(x_range)[:, 1]
    ax.plot(x_range, proba, color='#0d6efd', linewidth=2.5, label='Predicted Probability')
    ax.axhline(y=0.5, color='#6c757d', linestyle='--', linewidth=1.5, label='Classification Threshold (0.5)')

    ax.set_xlabel('Study Hours', fontsize=12, fontweight='bold')
    ax.set_ylabel('Class / Probability', fontsize=12, fontweight='bold')
    ax.set_title('Logistic Regression: Study Hours vs Exam Result', fontsize=13, fontweight='bold')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()