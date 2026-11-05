"""
Módulo de escalamiento Min-Max y validación de dominio.
Cumple con la convención de conservar la forma (N, d) donde las filas son ejemplos.
"""
import numpy as np


def escalar(X_raw, limites):
    """
    Escala la matriz de características crudas X_raw de forma (N, 4) usando
    normalización Min-Max con mínimos iguales a cero:
        x' = (x - 0) / (x_max - 0) = x / limites

    Parámetros:
        X_raw: np.ndarray de forma (N, d) con ejemplos en filas.
        limites: np.ndarray de forma (d,) con los máximos de cada variable.

    Retorna:
        np.ndarray de forma (N, d) con los valores escalados sin redondeo previo.
    """
    X_arr = np.asarray(X_raw, dtype=float)
    lim_arr = np.asarray(limites, dtype=float)
    if np.any(lim_arr == 0):
        raise ValueError("Los límites máximos no pueden ser cero (división por cero).")
    return X_arr / lim_arr


def validar_rango_dominio(X_raw, limites, estrategia="conservar"):
    """
    Política configurable para el manejo de valores fuera del rango [0, limites].
    Permite evaluar las tres estrategias analizadas en el Diseño del Clasificador:
      - 'conservar': mantiene la extrapolación lineal (ej. 150 km/h -> 1.25).
      - 'recortar': aplica clipping al intervalo [0, limites] (ej. 150 km/h -> 1.0).
      - 'rechazar': lanza una excepción de validación de datos si excede límites.
    """
    X_arr = np.asarray(X_raw, dtype=float)
    lim_arr = np.asarray(limites, dtype=float)
    if estrategia == "conservar":
        return escalar(X_arr, lim_arr)
    elif estrategia == "recortar":
        X_clipped = np.clip(X_arr, 0.0, lim_arr)
        return escalar(X_clipped, lim_arr)
    elif estrategia == "rechazar":
        if np.any(X_arr < 0.0) or np.any(X_arr > lim_arr):
            raise ValueError("Datos fuera de los límites físicos permitidos.")
        return escalar(X_arr, lim_arr)
    else:
        raise ValueError(f"Estrategia desconocida: {estrategia}")
