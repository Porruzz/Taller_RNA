"""Evaluación de la implementación del estudiante: python main.py."""
import numpy as np
from datos import X_RAW, Y, LIMITES, W1, W2, B1, B2
from red import escalar, propagar, clasificar, entropia_cruzada

def main():
    X = escalar(X_RAW, LIMITES).T
    Z1, A1, Z2, P = propagar(X, W1, B1, W2, B2)
    for nombre, matriz in [('X',X),('W1',W1),('B1',B1),
                           ('A1',A1),('W2',W2),('B2',B2),('P',P)]:
        print(nombre, matriz.shape)
    assert P.shape == (1, 7), 'Revisar forma de salida'
    assert np.allclose(P[0,:2], [.002046,.999962], atol=1e-6)
    individual = np.hstack([propagar(X[:,i:i+1],W1,B1,W2,B2)[3]
                            for i in range(X.shape[1])])
    assert np.allclose(P, individual), 'Lote e individual difieren'
    C = clasificar(P).ravel()
    print('\nRegistro | y | puntuación | clase | acierto')
    for i, (real, p, c) in enumerate(zip(Y, P.ravel(), C), 1):
        print(f'{i:8d} | {real} | {p:.6f} | {c} | {real == c}')
    print('Exactitud didáctica:', np.mean(C == Y))
    print('Entropía cruzada:', entropia_cruzada(Y, P))
    for t in [.3,.5,.8]:
        print(f'Umbral {t}:', clasificar(P,t).ravel())
    print('Verificaciones numéricas superadas.')

if __name__ == '__main__':
    main()
