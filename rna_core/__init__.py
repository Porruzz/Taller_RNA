"""
Paquete modular `rna_core`: Motor de Inferencia y Análisis de Redes Neuronales Artificiales.
Diseñado bajo principios de Arquitectura Limpia (Clean Architecture), Separación de
Responsabilidades (SoC) y Principio de Responsabilidad Única (SRP).
"""

from rna_core.activations.functions import sigmoide, escalon_heaviside
from rna_core.preprocessing.scaler import escalar, validar_rango_dominio
from rna_core.layers.propagation import propagar
from rna_core.decision.classifier import clasificar
from rna_core.evaluation.metrics import entropia_cruzada, exactitud

__all__ = [
    "sigmoide",
    "escalon_heaviside",
    "escalar",
    "validar_rango_dominio",
    "propagar",
    "clasificar",
    "entropia_cruzada",
    "exactitud",
]
