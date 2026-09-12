import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Datos de entrenamiento
costos = np.array([[5000], [10000], [15000], [20000], [25000]])
pesos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(costos, pesos)

# Calcular métricas
pendiente = float(modelo.coef_[0])
intercepto = float(modelo.intercept_)
r2_score = float(r2_score(pesos, modelo.predict(costos)))

# Datos para tabla
costos_flat = costos.flatten().tolist()
pesos_flat = pesos.tolist()
predicciones = modelo.predict(costos).flatten().tolist()

def calculateWeight(cost):
    """Calcula el peso estimado basado en el costo de envío"""
    if cost < 0:
        raise ValueError("El costo no puede ser negativo")
    prediccion = modelo.predict([[float(cost)]])
    return round(float(prediccion[0]), 2)

# Exportar variables
__all__ = ['calculateWeight', 'pendiente', 'intercepto', 'r2_score', 
           'costos_flat', 'pesos_flat', 'predicciones', 'modelo']