"""
Módulo analítico para la Sección 2: 'Propagación multicapa (Red 2-2-1)'
"""
import numpy as np
from rna_core.layers.propagation import propagar
from rna_core.decision.classifier import clasificar
from rna_core.evaluation.metrics import entropia_cruzada


def evaluar_red_221(b2_valor=0.0):
    """
    Evalúa paso a paso la red 2-2-1 del taller para x = (1, 0.5)^T y etiqueta y = 1:
        W1 = [[1, -1], [0.5, 0.5]], b1 = [[0], [0]]
        W2 = [[1, -1]], b2 = [[b2_valor]]
    """
    x = np.array([[1.0], [0.5]])
    y = np.array([1])
    W1 = np.array([[1.0, -1.0], [0.5, 0.5]])
    b1 = np.array([[0.0], [0.0]])
    W2 = np.array([[1.0, -1.0]])
    b2 = np.array([[float(b2_valor)]])

    z1, a1, z2, y_hat = propagar(x, W1, b1, W2, b2)
    clase = clasificar(y_hat, umbral=0.5)
    perdida = entropia_cruzada(y, y_hat)

    # Demostración de colapso afín sin activación oculta:
    # z2_lineal = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x + (W2 @ b1 + b2)
    W_efectivo = W2 @ W1
    b_efectivo = W2 @ b1 + b2
    z2_sin_activacion = float((W_efectivo @ x + b_efectivo).ravel()[0])

    return {
        "x": x.ravel(),
        "y": int(y[0]),
        "b2": float(b2_valor),
        "z1": z1.ravel(),
        "a1": a1.ravel(),
        "z2": float(z2.ravel()[0]),
        "y_hat": float(y_hat.ravel()[0]),
        "clase_05": int(clase.ravel()[0]),
        "perdida_bce": float(perdida),
        "W_efectivo_colapso": W_efectivo.ravel(),
        "b_efectivo_colapso": float(b_efectivo.ravel()[0]),
        "z2_sin_activacion": z2_sin_activacion,
    }
