import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)

DATA_PATH = os.path.join('data', 'ridge_classifier.csv')
FEATURES = ['study_hours', 'attendance', 'previous_grade', 'assignments_completed']


def load_ridge_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError('Ridge classifier dataset not found. Run data/generate_datasets.py')
    df = pd.read_csv(DATA_PATH).dropna()
    return df


def train_ridge_model():
    df = load_ridge_data()
    X = df[FEATURES].values
    y = df['academic_risk'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_train, y_train)
    info = {
        'records': int(len(df)),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'features': FEATURES,
        'target': 'academic_risk',
        'class_0_meaning': 'Low Risk',
        'class_1_meaning': 'High Risk'
    }
    return model, info


def predict_academic_risk(model, values):
    features = [float(v) for v in values]
    if any(v < 0 for v in features):
        raise ValueError('All values must be non-negative.')
    prediction = int(model.predict([features])[0])
    return prediction


def get_ridge_metrics(model):
    df = load_ridge_data()
    X = df[FEATURES].values
    y = df['academic_risk'].values
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


def generate_ridge_plot(model):
    df = load_ridge_data()
    low = df[df['academic_risk'] == 0]
    high = df[df['academic_risk'] == 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(low['study_hours'], low['previous_grade'], color='#198754',
               alpha=0.5, s=30, label='Low Risk (0)')
    ax.scatter(high['study_hours'], high['previous_grade'], color='#dc3545',
               alpha=0.5, s=30, label='High Risk (1)')
    ax.set_xlabel('Study Hours', fontsize=12, fontweight='bold')
    ax.set_ylabel('Previous Grade', fontsize=12, fontweight='bold')
    ax.set_title('Ridge Classifier: Academic Risk Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()