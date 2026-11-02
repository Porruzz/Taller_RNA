"""Complete estas funciones. Ejemplos como columnas en propagar."""
import numpy as np

def sigmoide(z):
    """TODO 1: activación elemento a elemento; evitar desbordamiento."""
    raise NotImplementedError('Implementar sigmoide')

def escalar(X_raw, limites):
    """TODO 2: recibe (N,4), devuelve (N,4); mínimos iguales a cero."""
    raise NotImplementedError('Implementar escalamiento')

def propagar(X, W1, b1, W2, b2):
    """TODO 3: X (4,N). Devuelve Z1, A1, Z2, P, en ese orden."""
    raise NotImplementedError('Implementar propagación')

def clasificar(probabilidades, umbral=0.5):
    """TODO 4: devolver enteros usando >=; conservar dimensiones."""
    raise NotImplementedError('Implementar clasificación')

def entropia_cruzada(y, probabilidades):
    """TODO 5: pérdida media; aplanar vectores y proteger log(0)."""
    raise NotImplementedError('Implementar entropía cruzada')
