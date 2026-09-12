from flask import Flask, render_template, request
import WeightRegression
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import os

from models.linear_regression import (
    train_linear_model, predict_package_weight, generate_linear_plot
)
from models.logistic_regression import (
    train_logistic_model, predict_exam_result, generate_logistic_plot,
    get_logistic_metrics
)
from models.ridge_classifier import (
    train_ridge_model, predict_academic_risk, generate_ridge_plot,
    get_ridge_metrics
)

app = Flask(__name__)

linear_model, linear_info = train_linear_model()
logistic_model, logistic_info = train_logistic_model()
ridge_model, ridge_info = train_ridge_model()


@app.route('/')
def home():
    plot_url = generate_plots()
    metrics = {
        'pendiente': WeightRegression.pendiente,
        'intercepto': WeightRegression.intercepto,
        'r2_score': WeightRegression.r2_score
    }
    data = {
        'costos': WeightRegression.costos_flat,
        'pesos': WeightRegression.pesos_flat,
        'predicciones': WeightRegression.predicciones
    }
    return render_template('index.html',
                           plot_url=plot_url,
                           metrics=metrics,
                           data=data)


@app.route("/weight/", methods=["POST"])
def predict_weight():
    import json
    try:
        cost = float(request.form.get("cost", 0))
        if cost < 0:
            result = {"error": "The shipping cost must be a positive value"}
        else:
            prediction = WeightRegression.calculateWeight(cost)
            result = {"success": True, "prediction": prediction, "cost": cost}
    except ValueError:
        result = {"error": "Please enter a valid numeric value"}
    except Exception as e:
        result = {"error": f"Calculation error: {str(e)}"}
    return json.dumps(result)


def generate_plots():
    costos_np = np.array(WeightRegression.costos_flat).reshape(-1, 1)
    pesos_np = np.array(WeightRegression.pesos_flat)
    pred_np = WeightRegression.modelo.predict(costos_np)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(costos_np, pesos_np, color='#f5576c', s=120,
                label='Actual Values', zorder=5, edgecolors='white', linewidth=2)
    ax1.plot(costos_np, pred_np, color='#4facfe', linewidth=3,
             label='Regression Line')
    ax1.set_xlabel('Shipping Cost (COP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Package Weight (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('Linear Regression: Shipping Cost vs Weight', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')

    residuos = pesos_np - pred_np.flatten()
    ax2.scatter(pred_np, residuos, color='#43e97b', s=120,
                zorder=5, edgecolors='white', linewidth=2)
    ax2.axhline(y=0, color='#f5576c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode()


@app.route('/machine-learning/concepts')
def ml_concepts():
    return render_template('machine_learning/concepts.html')


@app.route('/machine-learning/types')
def ml_types():
    return render_template('machine_learning/types.html')


@app.route('/use-cases/use-case-1')
def use_case_1():
    return render_template('use_cases/use_case1.html')


@app.route('/use-cases/use-case-2')
def use_case_2():
    return render_template('use_cases/use_case2.html')


@app.route('/use-cases/use-case-3')
def use_case_3():
    return render_template('use_cases/use_case3.html')


@app.route('/use-cases/use-case-4')
def use_case_4():
    return render_template('use_cases/use_case4.html')


@app.route('/linear-regression/concepts')
def linear_concepts():
    return render_template('linear_regression/concepts.html')


@app.route('/linear-regression/application', methods=['GET', 'POST'])
def linear_application():
    result = None
    error = None
    submitted_cost = None
    if request.method == 'POST':
        raw = request.form.get('cost', '').strip()
        if not raw:
            error = 'Please enter a shipping cost.'
        else:
            try:
                submitted_cost = float(raw)
                result = predict_package_weight(linear_model, submitted_cost)
            except ValueError as e:
                error = str(e) if str(e) else 'Please enter a valid number.'
    plot_url = generate_linear_plot(linear_model)
    return render_template(
        'linear_regression/application.html',
        info=linear_info, plot_url=plot_url,
        result=result, error=error, submitted_cost=submitted_cost
    )


@app.route('/logistic-regression/concepts')
def logistic_concepts():
    return render_template('logistic_regression/concepts.html')


@app.route('/logistic-regression/application', methods=['GET', 'POST'])
def logistic_application():
    result = None
    probability = None
    error = None
    submitted_value = None
    if request.method == 'POST':
        raw = request.form.get('study_hours', '').strip()
        if not raw:
            error = 'Please enter study hours.'
        else:
            try:
                submitted_value = float(raw)
                result, probability = predict_exam_result(logistic_model, submitted_value)
            except ValueError as e:
                error = str(e) if str(e) else 'Please enter a valid number.'
    plot_url = generate_logistic_plot(logistic_model)
    return render_template(
        'logistic_regression/application.html',
        info=logistic_info, plot_url=plot_url,
        result=result, probability=probability,
        error=error, submitted_value=submitted_value
    )


@app.route('/logistic-regression/metrics')
def logistic_metrics():
    metrics = get_logistic_metrics(logistic_model)
    return render_template('logistic_regression/metrics.html', metrics=metrics, info=logistic_info)


@app.route('/ridge-classifier/concepts')
def ridge_concepts():
    return render_template('ridge_classifier/concepts.html')


@app.route('/ridge-classifier/application', methods=['GET', 'POST'])
def ridge_application():
    result = None
    error = None
    submitted_values = None
    if request.method == 'POST':
        fields = ['study_hours', 'attendance', 'previous_grade', 'assignments_completed']
        raw_values = [request.form.get(f, '').strip() for f in fields]
        if any(v == '' for v in raw_values):
            error = 'Please fill in all fields.'
        else:
            try:
                submitted_values = {f: float(v) for f, v in zip(fields, raw_values)}
                result = predict_academic_risk(ridge_model, raw_values)
            except ValueError as e:
                error = str(e) if str(e) else 'Please enter valid numeric values.'
    plot_url = generate_ridge_plot(ridge_model)
    return render_template(
        'ridge_classifier/application.html',
        info=ridge_info, plot_url=plot_url,
        result=result, error=error, submitted_values=submitted_values
    )


@app.route('/ridge-classifier/metrics')
def ridge_metrics():
    metrics = get_ridge_metrics(ridge_model)
    return render_template('ridge_classifier/metrics.html', metrics=metrics, info=ridge_info)


@app.route('/comparison')
def comparison():
    logistic_metrics_data = get_logistic_metrics(logistic_model)
    ridge_metrics_data = get_ridge_metrics(ridge_model)
    if ridge_metrics_data['f1'] > logistic_metrics_data['f1']:
        best = 'Ridge Classifier'
    elif logistic_metrics_data['f1'] > ridge_metrics_data['f1']:
        best = 'Logistic Regression'
    else:
        best = 'Both models are tied'
    return render_template(
        'comparison.html',
        logistic=logistic_metrics_data,
        ridge=ridge_metrics_data,
        best=best
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    #app.py actividad 1 y 2 