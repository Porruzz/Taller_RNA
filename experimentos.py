"""Complete la generación de curvas y ejecute python experimentos.py."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datos import LIMITES, W1, W2, B1, B2
from red import escalar, propagar
from rna_core.analytics.sensitivity import generar_curva_sensibilidad


def curva(valores, columna, sesgo=0.0):
    """TODO: repetir [65,5,180,80000], sustituir una columna,
    escalar, transponer, propagar y devolver P aplanado.
    El sesgo recibido reemplaza B2, sin modificar los datos globales.
    """
    return generar_curva_sensibilidad(
        valores=valores,
        columna=columna,
        sesgo=sesgo,
        limites=LIMITES,
        W1=W1,
        B1=B1,
        W2=W2,
    )


def main():
    salida = Path(__file__).parent / 'resultados'
    salida.mkdir(exist_ok=True)
    v = np.linspace(0,150,301)
    fig, ax = plt.subplots()
    for b in [-2,0,2]:
        ax.plot(v, curva(v,0,b), label=f'Sesgo de salida = {b}')
    ax.axvline(120, color='gray', linestyle='--', label='Límite didáctico')
    ax.axhline(.5, color='black', linestyle=':', label='Umbral 0.5')
    ax.set(xlabel='Velocidad (km/h)', ylabel='Salida sigmoide', ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida/'velocidad.png', dpi=160)
    e = np.linspace(0,400000,301)
    fig2, ax2 = plt.subplots()
    ax2.plot(e, curva(e,3), label='Salida del modelo (sesgo = 0)')
    ax2.axhline(.5, color='black', linestyle=':', label='Umbral 0.5')
    ax2.set(xlabel='Experiencia (km)', ylabel='Salida sigmoide', ylim=(0,1))
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(salida/'experiencia.png', dpi=160)
    plt.close('all')
    print('Gráficas guardadas en', salida)


if __name__ == '__main__':
    main()
