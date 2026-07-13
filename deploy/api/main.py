"""
API de Predicción de Abandono de Clientes y Recomendación de Retención.

Proyecto: Predicción de Abandono de Clientes mediante Machine Learning
          y un Agente de Utilidad para la Retención de Clientes.

Esta API expone dos endpoints:
  - POST /api/predecir   -> probabilidad de abandono estimada por el modelo XGBoost.
  - POST /api/recomendar -> probabilidad de abandono + acción de retención recomendada
                            por el Agente de Utilidad.

Nota de diseño: el modelo final (XGBoost) no requiere escalado de variables,
por lo que esta API no depende de scikit-learn en tiempo de ejecución,
solo de xgboost, numpy y pandas. Esto mantiene el tamaño del paquete de
despliegue muy por debajo del límite de Vercel para funciones Python.
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# Carga del modelo y de las columnas de entrenamiento
# ============================================================
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(DIRECTORIO_ACTUAL, "modelo", "modelo_xgb_final.joblib")
RUTA_COLUMNAS = os.path.join(DIRECTORIO_ACTUAL, "modelo", "columnas_modelo.joblib")

try:
    modelo = joblib.load(RUTA_MODELO)
    columnas_modelo = joblib.load(RUTA_COLUMNAS)
except FileNotFoundError as error:
    raise RuntimeError(
        "No se encontraron los artefactos del modelo. Ejecuta la Sección 17 del "
        "notebook del proyecto, descarga 'modelo_xgb_final.joblib' y "
        "'columnas_modelo.joblib' desde Google Drive, y colócalos en api/modelo/."
    ) from error

# ============================================================
# Parámetros del Agente de Utilidad (idénticos a los del notebook, Sección 10.1)
# ============================================================
HORIZONTE_MESES = 12

ACCIONES = {
    "Sin accion": {
        "efectividad": 0.00,
        "costo": lambda cargo_mensual: 0.0,
    },
    "Descuento en la tarifa": {
        "efectividad": 0.35,
        "costo": lambda cargo_mensual: 0.20 * cargo_mensual * HORIZONTE_MESES,
    },
    "Mejora de plan sin costo adicional": {
        "efectividad": 0.45,
        "costo": lambda cargo_mensual: 120.0,
    },
    "Contacto proactivo": {
        "efectividad": 0.15,
        "costo": lambda cargo_mensual: 5.0,
    },
}


def calcular_utilidad(probabilidad_churn: float, cargo_mensual: float, accion: str) -> float:
    """Utilidad esperada U(a) = p * e(a) * V - c(a), ver informe, apartado 4.1.2."""
    valor_cliente = cargo_mensual * HORIZONTE_MESES
    efectividad = ACCIONES[accion]["efectividad"]
    costo = ACCIONES[accion]["costo"](cargo_mensual)
    return probabilidad_churn * efectividad * valor_cliente - costo


def recomendar_accion(probabilidad_churn: float, cargo_mensual: float):
    utilidades = {
        accion: calcular_utilidad(probabilidad_churn, cargo_mensual, accion)
        for accion in ACCIONES
    }
    mejor_accion = max(utilidades, key=utilidades.get)
    return mejor_accion, utilidades


# ============================================================
# Preprocesamiento: replica exactamente la codificación del notebook
# (Sección 5: mapeo binario + One-Hot con drop_first), Sección III del informe.
# ============================================================
VARIABLES_BINARIAS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
VARIABLES_NOMINALES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]


class ClienteInput(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(examples=[0], ge=0, le=1)
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[12], ge=0)
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["No"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["Yes"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[89.9], ge=0)
    TotalCharges: float = Field(examples=[1078.8], ge=0)


def preprocesar_cliente(cliente: ClienteInput) -> pd.DataFrame:
    df = pd.DataFrame([cliente.model_dump()])

    mapeo_binario = {"Yes": 1, "No": 0}
    for var in VARIABLES_BINARIAS:
        df[var] = df[var].map(mapeo_binario)
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(df, columns=VARIABLES_NOMINALES, drop_first=True)

    # Alinea las columnas exactamente con las usadas en el entrenamiento.
    # Cualquier categoría no vista o columna dummy ausente en este registro
    # (por tratarse de una sola fila) se completa con 0.
    df = df.reindex(columns=columnas_modelo, fill_value=0)

    if df.isnull().any().any():
        raise HTTPException(status_code=400, detail="Datos de cliente inválidos o incompletos.")

    return df


# ============================================================
# Aplicación FastAPI
# ============================================================
app = FastAPI(
    title="API de Predicción de Abandono de Clientes",
    description="Proyecto académico: Machine Learning + Agente de Utilidad para retención.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://churn-prediction-frontend-one.vercel.app",
        "http://localhost:3000",
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/api")
def raiz():
    return {
        "proyecto": "Predicción de Abandono de Clientes y Agente de Utilidad",
        "endpoints": ["/api/predecir", "/api/recomendar"],
    }


@app.post("/api/predecir")
def predecir(cliente: ClienteInput):
    X = preprocesar_cliente(cliente)
    probabilidad = float(modelo.predict_proba(X)[0, 1])
    return {"probabilidad_abandono": round(probabilidad, 4)}


@app.post("/api/recomendar")
def recomendar(cliente: ClienteInput):
    X = preprocesar_cliente(cliente)
    probabilidad = float(modelo.predict_proba(X)[0, 1])
    accion, utilidades = recomendar_accion(probabilidad, cliente.MonthlyCharges)
    return {
        "probabilidad_abandono": round(probabilidad, 4),
        "accion_recomendada": accion,
        "utilidad_por_accion": {k: round(v, 2) for k, v in utilidades.items()},
    }
