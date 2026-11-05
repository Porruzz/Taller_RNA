"""
Implementación de la capa de decisión por umbralización (Thresholding).
Separa la activación continua y_hat de la decisión discreta c_hat.
"""
import numpy as np


def clasificar(probabilidades, umbral=0.5):
    """
    Convierte puntuaciones o probabilidades continuas en etiquetas binarias {0, 1}
    aplicando la regla de decisión:
        c_hat = 1 si y_hat >= umbral, y 0 en otro caso.

    Conserva exactamente las dimensiones del arreglo de entrada y devuelve enteros.
    """
    probs = np.asarray(probabilidades, dtype=float)
    clases = (probs >= float(umbral)).astype(int)
    if np.ndim(probabilidades) == 0:
        return int(clases)
    return clases
