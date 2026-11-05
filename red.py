"""Complete estas funciones. Ejemplos como columnas en propagar.
Arquitectura Modular: Este módulo actúa como Fachada (Facade Pattern) delegando
las responsabilidades matemáticas a los submódulos desacoplados de `rna_core`.
"""
import numpy as np
from rna_core.activations.functions import sigmoide as _sigmoide_core
from rna_core.preprocessing.scaler import escalar as _escalar_core
from rna_core.layers.propagation import propagar as _propagar_core
from rna_core.decision.classifier import clasificar as _clasificar_core
from rna_core.evaluation.metrics import entropia_cruzada as _entropia_cruzada_core


def sigmoide(z):
    """TODO 1: activación elemento a elemento; evitar desbordamiento."""
    return _sigmoide_core(z)


def escalar(X_raw, limites):
    """TODO 2: recibe (N,4), devuelve (N,4); mínimos iguales a cero."""
    return _escalar_core(X_raw, limites)


def propagar(X, W1, b1, W2, b2):
    """TODO 3: X (4,N). Devuelve Z1, A1, Z2, P, en ese orden."""
    return _propagar_core(X, W1, b1, W2, b2)


def clasificar(probabilidades, umbral=0.5):
    """TODO 4: devolver enteros usando >=; conservar dimensiones."""
    return _clasificar_core(probabilidades, umbral=umbral)


def entropia_cruzada(y, probabilidades):
    """TODO 5: pérdida media; aplanar vectores y proteger log(0)."""
    return _entropia_cruzada_core(y, probabilidades)
