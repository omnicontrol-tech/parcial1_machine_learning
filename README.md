# Resistencia a la compresión del concreto – app en Streamlit

App web basada en el notebook `Concrete_compresive_strength.ipynb`. Ajustas las cantidades de la mezcla, eliges el modelo (regresión lineal o MLP) y te dice la resistencia estimada en MPa.

## Qué hay en la carpeta

| Archivo | Para qué sirve |
|---|---|
| `app.py` | La app: interfaz, selector de modelo, predicción y gráficas |
| `Concrete_Data.csv` | El dataset, convertido de `.xls` a `.csv` |
| `mlp_weights.npz` | Pesos del MLP ya entrenado (arquitectura 128-64-32-1 del notebook) |
| `train_mlp.py` | Script que entrena el MLP con Keras y genera `mlp_weights.npz` |
| `requirements.txt` | Librerías que necesita la app |

## Correrla en tu computador

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre sola en http://localhost:8501. En la barra lateral eliges el modelo y mueves los controles de la mezcla.

## Desplegarla gratis en Streamlit Community Cloud

1. Crea un repositorio público en GitHub y sube los cinco archivos de arriba (`app.py`, `Concrete_Data.csv`, `mlp_weights.npz`, `train_mlp.py` y `requirements.txt`).
2. Entra a https://share.streamlit.io e inicia sesión con GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Elige el repositorio, la rama `main` y `app.py` como archivo principal.
5. **Deploy**. En uno o dos minutos tienes una URL pública.

Cada `git push` con cambios actualiza la app sola.

## Cómo está armado

- **Normalización:** igual que en el notebook, todas las columnas (incluida la resistencia) van a [-1, 1]. La app normaliza lo que ingresa el usuario con el mínimo y máximo del dataset y desnormaliza la predicción para devolverla en MPa.
- **Regresión lineal:** se entrena al abrir la app (es instantáneo) y queda en caché.
- **MLP:** se entrena una vez con `train_mlp.py` y los pesos quedan guardados en `mlp_weights.npz`. La app solo calcula el paso hacia adelante con NumPy, así que **no hace falta instalar TensorFlow** para desplegar (que es pesado y a veces da problemas de versiones en la nube). Verifiqué que las predicciones coinciden con las de Keras (diferencia del orden de 1e-7).
- **Si cambias el MLP** (más épocas, otra arquitectura, otro optimizador), corre `pip install tensorflow` y luego `python train_mlp.py`. Eso regenera `mlp_weights.npz`. Si cambias la arquitectura, ajusta también `mlp_predict` en `app.py`.

## Resultados (sobre el 20 % de prueba)

| Modelo | R² | MAE (MPa) | RMSE (MPa) |
|---|---|---|---|
| Regresión lineal | 0,628 | 7,75 | 9,80 |
| MLP | 0,849 | 4,45 | 6,24 |

El MLP mejora bastante porque capta la relación no lineal entre la edad y la resistencia. Sus resultados cambian un poco cada vez que se reentrena, porque los pesos arrancan aleatorios (en `train_mlp.py` fijé la semilla para que sea repetible). Además, la pérdida de validación queda por encima de la de entrenamiento, así que hay algo de sobreajuste; añadir *early stopping* o *dropout* podría ayudar.
