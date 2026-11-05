"""
Implementación de funciones de activación con estabilidad numérica garantizada.
"""
import numpy as np


def sigmoide(z):
    """
    Calcula la función logística sigmoide elemento a elemento:
        sigma(z) = 1 / (1 + exp(-z))

    Evita desbordamientos (overflow en exp(-z) para z << 0) acotando el argumento
    en el rango seguro de punto flotante de doble precisión IEEE 754 [-500, 500].
    """
    z_arr = np.asarray(z, dtype=float)
    z_seguro = np.clip(z_arr, -500.0, 500.0)
    resultado = 1.0 / (1.0 + np.exp(-z_seguro))
    if np.ndim(z) == 0:
        return float(resultado)
    return resultado


def escalon_heaviside(z):
    """
    Función escalón unitario H(z) definida en el taller:
        H(z) = 1 si z >= 0, y 0 en otro caso.
    Utilizada para el reto opcional de la compuerta XOR.
    """
    z_arr = np.asarray(z, dtype=float)
    resultado = (z_arr >= 0.0).astype(int)
    if np.ndim(z) == 0:
        return int(resultado)
    return resultado
