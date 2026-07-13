"""
Pruebas de humo (smoke tests) para la API de predicción de abandono.

Ejecutar con: pytest tests/test_api.py -v
Requiere que api/modelo/ contenga modelo_xgb_final.joblib y columnas_modelo.joblib
(ver api/modelo/COLOCAR_MODELO_AQUI.md).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy", "api"))

import pytest
from fastapi.testclient import TestClient
from main import app

cliente_prueba = TestClient(app)

CLIENTE_EJEMPLO = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 477.5,
}


def test_raiz_responde():
    respuesta = cliente_prueba.get("/api")
    assert respuesta.status_code == 200
    assert "endpoints" in respuesta.json()


def test_predecir_devuelve_probabilidad_valida():
    respuesta = cliente_prueba.post("/api/predecir", json=CLIENTE_EJEMPLO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "probabilidad_abandono" in cuerpo
    assert 0.0 <= cuerpo["probabilidad_abandono"] <= 1.0


def test_recomendar_devuelve_accion_valida():
    respuesta = cliente_prueba.post("/api/recomendar", json=CLIENTE_EJEMPLO)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert "accion_recomendada" in cuerpo
    assert "utilidad_por_accion" in cuerpo
    assert len(cuerpo["utilidad_por_accion"]) == 4
    # La acción recomendada debe ser, por definición, la de mayor utilidad esperada.
    mejor_segun_tabla = max(cuerpo["utilidad_por_accion"], key=cuerpo["utilidad_por_accion"].get)
    assert cuerpo["accion_recomendada"] == mejor_segun_tabla


def test_cliente_bajo_riesgo_recomienda_sin_accion():
    cliente_bajo_riesgo = dict(CLIENTE_EJEMPLO)
    cliente_bajo_riesgo.update({
        "tenure": 65,
        "Contract": "Two year",
        "MonthlyCharges": 25.0,
        "TotalCharges": 1625.0,
        "OnlineSecurity": "Yes",
        "TechSupport": "Yes",
    })
    respuesta = cliente_prueba.post("/api/recomendar", json=cliente_bajo_riesgo)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    # Con probabilidad de abandono muy baja, ninguna acción paga su propio costo.
    assert cuerpo["probabilidad_abandono"] < 0.3
