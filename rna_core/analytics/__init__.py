"""Módulo de analítica avanzada, sensibilidad y resolución de casos de estudio."""
from rna_core.analytics.sensitivity import generar_curva_sensibilidad
from rna_core.analytics.single_neuron import evaluar_neurona_apartamentos
from rna_core.analytics.mlp_221 import evaluar_red_221
from rna_core.analytics.xor_solver import evaluar_xor_heaviside

__all__ = [
    "generar_curva_sensibilidad",
    "evaluar_neurona_apartamentos",
    "evaluar_red_221",
    "evaluar_xor_heaviside",
]
