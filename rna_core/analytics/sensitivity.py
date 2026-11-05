"""
Módulo de análisis de sensibilidad univariado (Ceteris Paribus).
Permite variar una característica manteniendo constantes las demás entradas
del registro base sin mutar el estado global.
"""
import numpy as np
from rna_core.preprocessing.scaler import escalar
from rna_core.layers.propagation import propagar


REGISTRO_BASE_DEFECTO = np.array([65.0, 5.0, 180.0, 80000.0], dtype=float)


def generar_curva_sensibilidad(
    valores,
    columna,
    sesgo=0.0,
    limites=None,
    W1=None,
    B1=None,
    W2=None,
    registro_base=None,
):
    """
    Repite el registro base [65, 5, 180, 80000], sustituye la columna indicada
    con el vector `valores`, escala, transpone, propaga con el sesgo de salida
    especificado (sin alterar B2 global) y devuelve el vector P aplanado.
    """
    valores_arr = np.asarray(valores, dtype=float).ravel()
    base = (
        np.asarray(registro_base, dtype=float)
        if registro_base is not None
        else REGISTRO_BASE_DEFECTO.copy()
    )

    # Crear copia local (M, 4) sin modificar datos globales
    X_lote_raw = np.tile(base, (valores_arr.size, 1))
    X_lote_raw[:, int(columna)] = valores_arr

    # Escalar conservando forma (M, 4) y luego transponer a (4, M)
    X_escalado = escalar(X_lote_raw, limites).T

    # Crear sesgo de salida local sin modificar B2 global
    b2_local = np.array([[float(sesgo)]], dtype=float)

    _, _, _, P = propagar(X_escalado, W1, B1, W2, b2_local)
    return P.ravel()
