"""
Módulo analítico para el Reto Opcional: Compuerta XOR con Escalón de Heaviside.
"""
import numpy as np
from rna_core.activations.functions import escalon_heaviside


def evaluar_xor_heaviside():
    """
    Evalúa las cuatro combinaciones binarias de la compuerta XOR usando:
        h1 = H(x1 + x2 - 0.5)   [Equivalente a OR]
        h2 = H(x1 + x2 - 1.5)   [Equivalente a AND]
        y_hat = H(h1 - 2*h2 - 0.5)
    donde H(z) = 1 si z >= 0 y 0 en otro caso.
    """
    entradas = [(0, 0), (0, 1), (1, 0), (1, 1)]
    resultados = []
    for x1, x2 in entradas:
        z_h1 = x1 + x2 - 0.5
        h1 = escalon_heaviside(z_h1)

        z_h2 = x1 + x2 - 1.5
        h2 = escalon_heaviside(z_h2)

        z_out = h1 - 2 * h2 - 0.5
        y_hat = escalon_heaviside(z_out)
        xor_real = x1 ^ x2

        resultados.append(
            {
                "x1": x1,
                "x2": x2,
                "z_h1": z_h1,
                "h1": h1,
                "z_h2": z_h2,
                "h2": h2,
                "z_out": z_out,
                "y_hat": y_hat,
                "xor_real": xor_real,
                "correcto": (y_hat == xor_real),
            }
        )
    return resultados
