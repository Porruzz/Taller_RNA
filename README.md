# Taller de redes neuronales

Proyecto para estudiantes: propagación hacia adelante, sin entrenamiento.
Los siete registros son didácticos; no validan un sistema de seguridad vial.

## Instalación

Requiere Python 3.10 o superior. Desde esta carpeta:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell (alternativa): .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Secuencia de trabajo

1. Resolver los componentes matemático y de diseño del taller adjunto.
2. Completar los cinco TODO de `red.py`, en orden.
3. Ejecutar `python main.py` y revisar las verificaciones y resultados.
4. Completar `curva` en `experimentos.py`.
5. Ejecutar `python experimentos.py`. Revisar las imágenes en `resultados/`.
6. Elaborar el informe. Los errores NotImplementedError son intencionales
   hasta completar las funciones. No sustituirlos por salidas constantes.

## Convenciones

`X_RAW`: ejemplos en filas (7,4). `escalar` conserva esta forma.
`propagar`: ejemplos en columnas (4,N); salidas (1,N).
Pesos y sesgos permanecen fijos durante la predicción.
Los límites de escalamiento son los supuestos didácticos de la presentación.
No redondear las entradas antes de operar. Usar NumPy, sin redes automáticas.

## Entrega

Informe PDF, código completado y gráficas. Justificar resultados, umbrales,
limitaciones de los datos y decisiones de diseño. Dependencias requieren
acceso a un índice de paquetes o instalación previa.
