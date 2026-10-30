"""
App de Streamlit: predicción de la resistencia a la compresión del concreto.

Sigue el flujo del notebook (Concrete_compresive_strength.ipynb):
  1. Leer el dataset
  2. Normalizar todas las columnas al rango [-1, 1]
  3. Split 80/20 con random_state=42
  4. Dos modelos: regresión lineal y MLP (128-64-32-1)
  5. Predecir

El usuario elige el modelo en la barra lateral y mete las cantidades de la
mezcla; la app devuelve la resistencia estimada en MPa.

El MLP se entrenó con train_mlp.py (Keras) y sus pesos están en
mlp_weights.npz. Aquí solo se aplica con NumPy, así que no hace falta
instalar TensorFlow para desplegar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Resistencia del concreto", page_icon="🧱", layout="wide")

AQUI = Path(__file__).parent
DATA_PATH = AQUI / "Concrete_Data.csv"
MLP_PATH = AQUI / "mlp_weights.npz"
TARGET = "Concrete compressive"

LINEAL = "Regresión lineal"
MLP = "MLP (red neuronal)"

# Nombres de columnas (como en el notebook) -> etiqueta y unidad para la interfaz
ETIQUETAS = {
    "Cement": ("Cemento", "kg/m³"),
    "Blast Furnace Slag": ("Escoria de alto horno", "kg/m³"),
    "Fly Ash": ("Ceniza volante", "kg/m³"),
    "Water": ("Agua", "kg/m³"),
    "Superplasticizer": ("Superplastificante", "kg/m³"),
    "Coarse Aggregate": ("Agregado grueso", "kg/m³"),
    "Fine Aggregate": ("Agregado fino", "kg/m³"),
    "Age": ("Edad", "días"),
}


# --------------------------------------------------------------------------
# Lógica del notebook
# --------------------------------------------------------------------------
def normalizar(x, ymax, ymin, xmax, xmin):
    """y = m*x + b  (misma función del notebook)."""
    m = (ymax - ymin) / (xmax - xmin)
    b = ymin - m * xmin
    return m * x + b


def desnormalizar(y, ymax, ymin, xmax, xmin):
    """Inversa de normalizar: pasa de [-1, 1] a la escala original."""
    m = (ymax - ymin) / (xmax - xmin)
    b = ymin - m * xmin
    return (y - b) / m


def mlp_predict(pesos, X):
    """Forward pass del MLP: Dense(128,relu) -> Dense(64,relu) -> Dense(32,relu) -> Dense(1)."""
    a = np.asarray(X, dtype=np.float64)
    for i in range(4):
        a = a @ pesos[f"W{i}"] + pesos[f"b{i}"]
        if i < 3:
            a = np.maximum(a, 0.0)  # ReLU
    return a.ravel()


@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def preparar_modelos():
    """Entrena la regresión lineal, carga el MLP y calcula métricas de ambos."""
    data = cargar_datos()
    dmax, dmin = data.max(), data.min()

    data2 = normalizar(data, 1, -1, dmax, dmin)
    X = data2.drop(TARGET, axis=1)
    y = data2[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lineal = linear_model.LinearRegression().fit(X_train, y_train)

    mlp_pesos = None
    if MLP_PATH.exists():
        archivo = np.load(MLP_PATH)
        mlp_pesos = {k: archivo[k] for k in archivo.files}

    def a_mpa(v):
        return desnormalizar(np.asarray(v), 1, -1, dmax[TARGET], dmin[TARGET])

    y_real = a_mpa(y_test)
    predictores = {LINEAL: lambda Xn: lineal.predict(Xn)}
    if mlp_pesos is not None:
        predictores[MLP] = lambda Xn: mlp_predict(mlp_pesos, Xn)

    resultados = {}
    for nombre, fn in predictores.items():
        y_pred = a_mpa(fn(X_test))
        resultados[nombre] = {
            "y_pred": y_pred,
            "R²": r2_score(y_real, y_pred),
            "MAE": mean_absolute_error(y_real, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        }

    return {
        "lineal": lineal,
        "mlp_pesos": mlp_pesos,
        "predictores": predictores,
        "dmax": dmax,
        "dmin": dmin,
        "columnas": X.columns.tolist(),
        "y_real": y_real,
        "resultados": resultados,
    }


def predecir(m, nombre_modelo, valores: dict) -> float:
    """Recibe la mezcla en unidades reales y devuelve MPa con el modelo elegido."""
    cols = m["columnas"]
    fila = pd.DataFrame([valores])[cols]
    fila_norm = normalizar(fila, 1, -1, m["dmax"][cols], m["dmin"][cols])
    pred_norm = m["predictores"][nombre_modelo](fila_norm)[0]
    return float(
        desnormalizar(pred_norm, 1, -1, m["dmax"][TARGET], m["dmin"][TARGET])
    )


# --------------------------------------------------------------------------
# Interfaz
# --------------------------------------------------------------------------
data = cargar_datos()
m = preparar_modelos()
modelos_disponibles = list(m["predictores"].keys())
st.info("Profesor Andres Felipe Escobar")
st.title("🧱 Resistencia a la compresión del concreto")
st.caption(
    "Estimación con modelos entrenados sobre el dataset Concrete Compressive "
    "Strength (1.030 mezclas)."
)

# ---- Barra lateral: modelo + mezcla ---------------------------------------
st.sidebar.header("Modelo")
modelo_elegido = st.sidebar.radio(
    "Selecciona el modelo",
    modelos_disponibles,
    help="Regresión lineal: sencillo e interpretable. MLP: red neuronal, capta relaciones no lineales.",
)
if m["mlp_pesos"] is None:
    st.sidebar.warning("No se encontró mlp_weights.npz; solo está disponible la regresión lineal.")

st.sidebar.divider()
st.sidebar.header("Mezcla de concreto")
st.sidebar.caption("Ajusta las cantidades por m³ de mezcla.")

valores = {}
for col, (nombre, unidad) in ETIQUETAS.items():
    lo, hi = float(data[col].min()), float(data[col].max())
    medio = float(data[col].median())
    paso = 1.0 if col == "Age" else 0.5
    valores[col] = st.sidebar.slider(f"{nombre} ({unidad})", lo, hi, medio, step=paso)

tab_pred, tab_modelo, tab_datos = st.tabs(["Predicción", "Modelo", "Datos"])

# ---- Pestaña 1: predicción ------------------------------------------------
with tab_pred:
    metr = m["resultados"][modelo_elegido]
    resistencia = predecir(m, modelo_elegido, valores)

    st.markdown(f"Modelo seleccionado: **{modelo_elegido}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resistencia estimada", f"{resistencia:.1f} MPa")
    maximo = data[TARGET].max()
    c2.metric(
        "Máximo del dataset",
        f"{maximo:.1f} MPa",
        delta=f"{resistencia - maximo:+.1f} MPa",
    )
    c3.metric("Error típico del modelo (MAE)", f"± {metr['MAE']:.1f} MPa")

    if resistencia < 0:
        st.warning(
            "El modelo dio un valor negativo, que no tiene sentido físico. "
            "Pasa con la regresión lineal en combinaciones poco comunes de ingredientes."
        )

    # Comparación: lo que dice cada modelo para la misma mezcla
    if len(modelos_disponibles) > 1:
        st.subheader("Qué dice cada modelo para esta mezcla")
        comparacion = pd.DataFrame(
            {
                "Modelo": modelos_disponibles,
                "Resistencia estimada (MPa)": [
                    round(predecir(m, n, valores), 1) for n in modelos_disponibles
                ],
                "MAE en prueba (MPa)": [
                    round(m["resultados"][n]["MAE"], 2) for n in modelos_disponibles
                ],
            }
        )
        st.dataframe(comparacion, hide_index=True, width="stretch")

    # Dónde cae la predicción dentro de la distribución real
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(data[TARGET], bins=30, color="lightsteelblue", ax=ax)
    ax.axvline(resistencia, color="firebrick", linestyle="--", linewidth=2,
               label=f"Tu mezcla: {resistencia:.1f} MPa")
    ax.set_xlabel("Resistencia a la compresión (MPa)")
    ax.set_ylabel("Cantidad de mezclas")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    st.pyplot(fig)

    with st.expander("Ver la mezcla que ingresaste"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Ingrediente": [ETIQUETAS[c][0] for c in valores],
                    "Valor": list(valores.values()),
                    "Unidad": [ETIQUETAS[c][1] for c in valores],
                }
            ),
            hide_index=True,
            width="stretch",
        )

# ---- Pestaña 2: modelo ----------------------------------------------------
with tab_modelo:
    st.subheader(f"Desempeño: {modelo_elegido}")
    st.caption("Métricas sobre el 20 % de datos que el modelo no usó para entrenar.")
    a, b, c = st.columns(3)
    a.metric("R²", f"{metr['R²']:.3f}")
    b.metric("MAE", f"{metr['MAE']:.2f} MPa")
    c.metric("RMSE", f"{metr['RMSE']:.2f} MPa")

    col_izq, col_der = st.columns(2)

    with col_izq:
        if modelo_elegido == LINEAL:
            coef_df = pd.DataFrame(
                {
                    "Variable": [ETIQUETAS[c][0] for c in m["columnas"]],
                    "Coeficiente": m["lineal"].coef_,
                }
            ).sort_values("Coeficiente", ascending=False)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x="Coeficiente", y="Variable", data=coef_df, hue="Variable",
                        palette="viridis", legend=False, ax=ax)
            ax.set_title("Coeficientes de la regresión lineal", fontweight="bold")
            ax.grid(True, linestyle=":", alpha=0.6)
        else:
            archivo = np.load(MLP_PATH)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(archivo["loss"], label="Entrenamiento", color="royalblue", linewidth=2)
            ax.plot(archivo["val_loss"], label="Validación", color="orange", linewidth=2)
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Pérdida (MSE, datos normalizados)")
            ax.set_title("Pérdida del MLP por época", fontweight="bold")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

    with col_der:
        y_pred = metr["y_pred"]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(m["y_real"], y_pred, color="royalblue", alpha=0.6,
                   edgecolors="w", s=50, label="Predicciones")
        lims = [min(m["y_real"].min(), y_pred.min()), max(m["y_real"].max(), y_pred.max())]
        ax.plot(lims, lims, color="firebrick", linestyle="--", linewidth=2,
                label="Ajuste perfecto")
        ax.set_xlabel("Valores reales (MPa)")
        ax.set_ylabel("Predicciones (MPa)")
        ax.set_title("Reales vs. predicciones", fontweight="bold")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)

    if modelo_elegido == LINEAL:
        st.info(
            "Los coeficientes están calculados con las variables normalizadas entre -1 y 1, "
            "así que se pueden comparar entre sí: cuanto más grande el valor absoluto, "
            "más pesa esa variable en la predicción."
        )
    else:
        st.info(
            "El MLP tiene tres capas ocultas (128, 64 y 32 neuronas con ReLU) y una salida. "
            "Si la pérdida de validación se queda por encima de la de entrenamiento, "
            "hay algo de sobreajuste."
        )

    st.subheader("Comparación de modelos")
    tabla = pd.DataFrame(
        {
            "Modelo": modelos_disponibles,
            "R²": [round(m["resultados"][n]["R²"], 3) for n in modelos_disponibles],
            "MAE (MPa)": [round(m["resultados"][n]["MAE"], 2) for n in modelos_disponibles],
            "RMSE (MPa)": [round(m["resultados"][n]["RMSE"], 2) for n in modelos_disponibles],
        }
    )
    st.dataframe(tabla, hide_index=True, width="stretch")

# ---- Pestaña 3: datos -----------------------------------------------------
with tab_datos:
    st.subheader("Dataset")
    st.write(f"{data.shape[0]} filas y {data.shape[1]} columnas.")
    st.dataframe(data, width="stretch", height=300)

    st.subheader("Estadísticos descriptivos")
    st.dataframe(data.describe().T, width="stretch")

    st.subheader("Mapa de correlación de Pearson")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data.corr(method="pearson"), annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
