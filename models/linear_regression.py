import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = os.path.join('data', 'linear_regression.csv')


def load_linear_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError('Linear regression dataset not found. Run data/generate_datasets.py')
    df = pd.read_csv(DATA_PATH).dropna()
    return df


def train_linear_model():
    df = load_linear_data()
    X = df[['shipping_cost']].values
    y = df['package_weight'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'records': int(len(df)),
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'slope': round(float(model.coef_[0]), 6),
        'intercept': round(float(model.intercept_), 4),
        'r2': round(float(r2_score(y_test, y_pred)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    }
    return model, metrics


def predict_package_weight(model, cost):
    value = float(cost)
    if value <= 0:
        raise ValueError('Shipping cost must be a positive number.')
    prediction = model.predict([[value]])[0]
    return round(float(prediction), 3)


def generate_linear_plot(model):
    df = load_linear_data()
    X = df[['shipping_cost']].values
    y = df['package_weight'].values
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_range)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(X, y, color='#0d6efd', alpha=0.5, s=20, label='Actual Values')
    ax1.plot(x_range, y_line, color='#dc3545', linewidth=2.5, label='Regression Line')
    ax1.set_xlabel('Shipping Cost (COP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Package Weight (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Regression: Shipping Cost vs Weight', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    y_pred_all = model.predict(X)
    residuals = y - y_pred_all
    ax2.scatter(y_pred_all, residuals, color='#198754', alpha=0.5, s=20)
    ax2.axhline(y=0, color='#dc3545', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Weight (kg)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()