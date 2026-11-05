"""
Funciones de evaluación de desempeño y pérdida para clasificación binaria.
"""
import numpy as np


def entropia_cruzada(y, probabilidades):
    """
    Calcula la pérdida media de entropía cruzada binaria (Binary Cross-Entropy):
        L = - (1/N) * sum( y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) )

    Aplana ambos vectores de entrada y protege los logaritmos frente a log(0)
    acotando las probabilidades al intervalo [eps, 1 - eps].
    """
    y_flat = np.asarray(y, dtype=float).ravel()
    p_flat = np.asarray(probabilidades, dtype=float).ravel()

    eps = 1e-15
    p_seguro = np.clip(p_flat, eps, 1.0 - eps)

    perdida = -(y_flat * np.log(p_seguro) + (1.0 - y_flat) * np.log(1.0 - p_seguro))
    return float(np.mean(perdida))


def exactitud(y_real, y_pred):
    """
    Calcula la fracción de predicciones correctas (Accuracy):
        Acc = (1/N) * sum( I(y_real == y_pred) )
    """
    y_r = np.asarray(y_real, dtype=int).ravel()
    y_p = np.asarray(y_pred, dtype=int).ravel()
    return float(np.mean(y_r == y_p))
