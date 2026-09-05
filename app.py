from flask import Flask, render_template, request
import WeightRegression
import io
import base64
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    """Página principal con todo el contenido integrado"""
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
            result = {"error": "El costo debe ser un valor positivo"}
        else:
            prediction = WeightRegression.calculateWeight(cost)
            result = {"success": True, "prediction": prediction, "cost": cost}
    except ValueError:
        result = {"error": "Por favor, ingrese un valor numérico válido"}
    except Exception as e:
        result = {"error": f"Error en el cálculo: {str(e)}"}
    return json.dumps(result)

def generate_plots():
    costos_np = np.array(WeightRegression.costos_flat).reshape(-1, 1)
    pesos_np = np.array(WeightRegression.pesos_flat)
    pred_np = WeightRegression.modelo.predict(costos_np)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(costos_np, pesos_np, color='#f5576c', s=120, 
                label='Datos reales', zorder=5, edgecolors='white', linewidth=2)
    ax1.plot(costos_np, pred_np, color='#4facfe', linewidth=3, 
             label='Línea de regresión')
    ax1.set_xlabel('Costo de envío (COP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peso (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('Regresión Lineal: Costo vs Peso', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    residuos = pesos_np - pred_np.flatten()
    ax2.scatter(pred_np, residuos, color='#43e97b', s=120, 
                zorder=5, edgecolors='white', linewidth=2)
    ax2.axhline(y=0, color='#f5576c', linestyle='--', linewidth=2)
    ax2.set_xlabel('Valores predichos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuos', fontsize=12, fontweight='bold')
    ax2.set_title('Análisis de Residuos', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    img.seek(0)
    plt.close()
    
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)