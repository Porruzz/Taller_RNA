"""
Módulo analítico para la Sección 1: 'Una neurona que toma decisiones'
Caso de estudio: Alquiler de apartamentos según Precio y Área.
"""
import numpy as np
from rna_core.activations.functions import sigmoide
from rna_core.decision.classifier import clasificar


CASOS_APARTAMENTOS = [
    {"caso": "A", "precio": 4000.0, "area": 60.0},
    {"caso": "B", "precio": 6000.0, "area": 40.0},
    {"caso": "C", "precio": 2000.0, "area": 20.0},
    {"caso": "D", "precio": 7000.0, "area": 80.0},
]


def evaluar_neurona_apartamentos(w1=-2.0, w2=3.0, b=-0.5):
    """
    Evalúa los 4 casos del taller (A, B, C, D) con la neurona de decisión:
        x1 = precio / 8000
        x2 = area / 80
        z  = w1*x1 + w2*x2 + b
        y_hat = sigma(z)
    Calcula clasificaciones tanto para tau = 0.5 como para tau = 0.8.
    """
    resultados = []
    for item in CASOS_APARTAMENTOS:
        x1 = item["precio"] / 8000.0
        x2 = item["area"] / 80.0
        z = w1 * x1 + w2 * x2 + b
        y_hat = sigmoide(z)
        c_05 = clasificar(y_hat, umbral=0.5)
        c_08 = clasificar(y_hat, umbral=0.8)
        resultados.append(
            {
                "caso": item["caso"],
                "precio": item["precio"],
                "area": item["area"],
                "x1": x1,
                "x2": x2,
                "z": z,
                "y_hat": y_hat,
                "c_05": c_05,
                "c_08": c_08,
            }
        )
    return resultados


def frontera_decision_x2(x1, w1=-2.0, w2=3.0, b=-0.5):
    """
    Despeja x2 en la ecuación de frontera lineal w1*x1 + w2*x2 + b = 0:
        x2 = (-w1*x1 - b) / w2
    Para w1 = -2, w2 = 3, b = -0.5:
        x2 = (2/3)*x1 + 1/6
    Para w1 = -2, w2 = 3, b = +0.5:
        x2 = (2/3)*x1 - 1/6
    """
    x1_arr = np.asarray(x1, dtype=float)
    return (-w1 * x1_arr - b) / w2
