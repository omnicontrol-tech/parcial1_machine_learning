"""
Entrena el MLP del notebook (Keras) y guarda sus pesos en mlp_weights.npz.

Se corre una sola vez en local (necesita TensorFlow):
    pip install tensorflow
    python train_mlp.py

La app de Streamlit lee mlp_weights.npz y calcula la predicción con NumPy,
así que al desplegar NO hace falta instalar TensorFlow.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

SEED = 42
tf.keras.utils.set_random_seed(SEED)

TARGET = "Concrete compressive"
AQUI = Path(__file__).parent

data = pd.read_csv(AQUI / "Concrete_Data.csv")
dmax, dmin = data.max(), data.min()

# Misma normalización del notebook: cada columna a [-1, 1]
data2 = 2 * (data - dmin) / (dmax - dmin) - 1
X = data2.drop(TARGET, axis=1)
y = data2[TARGET]

# Mismo split del notebook
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Misma arquitectura del notebook: 128 -> 64 -> 32 -> 1
modelo = Sequential(
    [
        Input(shape=(X.shape[1],)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ]
)
modelo.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

history = modelo.fit(
    X_train, y_train,
    epochs=100, batch_size=32, verbose=2,
    validation_data=(X_test, y_test), shuffle=True,
)

pesos = {}
for i, capa in enumerate(modelo.layers):
    W, b = capa.get_weights()
    pesos[f"W{i}"], pesos[f"b{i}"] = W, b

np.savez(
    AQUI / "mlp_weights.npz",
    loss=np.array(history.history["loss"]),
    val_loss=np.array(history.history["val_loss"]),
    **pesos,
)
print("Pesos guardados en mlp_weights.npz")
