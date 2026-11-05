"""
Implementación de la propagación hacia adelante (Forward Propagation) matricial.
Opera tanto para un único vector columna (d, 1) como para un lote de N columnas (d, N).
"""
import numpy as np
from rna_core.activations.functions import sigmoide


def propagar(X, W1, b1, W2, b2):
    """
    Ejecuta la propagación hacia adelante en una red neuronal de dos capas (oculta + salida)
    con activación sigmoide en ambas capas:
        Z1 = W1 @ X + b1
        A1 = sigmoide(Z1)
        Z2 = W2 @ A1 + b2
        P  = sigmoide(Z2)

    Parámetros:
        X  : np.ndarray de forma (d, N) con ejemplos como columnas.
        W1 : np.ndarray de forma (h, d) con pesos de la capa oculta.
        b1 : np.ndarray de forma (h, 1) con sesgos de la capa oculta.
        W2 : np.ndarray de forma (k, h) con pesos de la capa de salida.
        b2 : np.ndarray de forma (k, 1) o escalar con sesgo de la capa de salida.

    Retorna:
        Tupla (Z1, A1, Z2, P) donde P tiene forma (1, N).
    """
    X_mat = np.asarray(X, dtype=float)
    if X_mat.ndim == 1:
        X_mat = X_mat.reshape(-1, 1)

    W1_mat = np.asarray(W1, dtype=float)
    b1_mat = np.asarray(b1, dtype=float).reshape(-1, 1)
    W2_mat = np.asarray(W2, dtype=float)
    b2_mat = np.asarray(b2, dtype=float).reshape(-1, 1)

    Z1 = W1_mat @ X_mat + b1_mat
    A1 = sigmoide(Z1)
    Z2 = W2_mat @ A1 + b2_mat
    P = sigmoide(Z2)

    return Z1, A1, Z2, P
