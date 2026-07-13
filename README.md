# Predicción de Abandono de Clientes — Machine Learning + Agente de Utilidad

Proyecto académico

---

## Descripción

Sistema de IA que predice la probabilidad de abandono (churn) de clientes en empresas de telecomunicaciones utilizando Machine Learning supervisado (Regresión Logística, Random Forest, XGBoost) y un agente racional basado en utilidad que recomienda la mejor estrategia de retención para cada cliente.

## Enlaces

- **Frontend desplegado:** [churn-prediction-frontend-one.vercel.app](https://churn-prediction-frontend-one.vercel.app)
- **API desplegada:** [churn-api-alpha-five.vercel.app](https://churn-api-alpha-five.vercel.app)
- **Frontend (repositorio):** [github.com/alehpr22-source/churn-frontend](https://github.com/alehpr22-source/churn-frontend)
- **Informe completo:** disponible en el repositorio local

---

## Dataset

- **Fuente:** [Telco Customer Churn — IBM / Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Registros:** 7,043 clientes
- **Variables:** 21 (demográficas, servicios contratados, datos de cuenta)
- **Variable objetivo:** `Churn` (Yes/No) — 26.5% de abandono

---

## Resultados

### Línea base (Accuracy)

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Regresión Logística | 73.8% | 50.4% | 78.3% | 61.4% | 0.842 |
| Random Forest | 78.8% | 63.0% | 48.7% | 54.9% | 0.829 |
| XGBoost | 75.7% | 53.6% | 61.8% | 57.4% | 0.812 |

La línea base reveló que ningún modelo superaba el 70% de Recall, umbral mínimo definido como requisito del proyecto.

### Después de optimizar por F2-score

El F2-score pondera el Recall el doble que la Precision (β=2), reflejando el costo asimétrico de los falsos negativos.

| Modelo | Accuracy | Precision | Recall | F1-score | F2-score | ROC-AUC |
|--------|----------|-----------|--------|----------|----------|---------|
| Regresión Logística | 73.8% | 50.4% | 78.3% | 61.4% | 70.5% | 0.842 |
| Random Forest | 76.7% | 54.4% | 75.7% | 63.3% | 70.2% | 0.844 |
| **XGBoost (seleccionado)** | **74.7%** | **51.5%** | **81.6%** | **63.1%** | **73.0%** | **0.846** |

**Mejor F2-score en validación cruzada:** XGBoost 0.733

**XGBoost fue seleccionado** por alcanzar el mayor F2-score (73.0%), Recall (81.6%) y ROC-AUC (0.846) entre los tres modelos.

---

## Arquitectura

```
Cliente (atributos) → Modelo XGBoost → Probabilidad de abandono
                                              |
                                              ↓
                                    Agente de Utilidad
                                 (evalúa 4 acciones de retención)
                                              |
                                              ↓
                      Acción de retención con mayor utilidad esperada
```

Las 4 acciones evaluadas por el agente son: Sin acción, Descuento en la tarifa (20%), Mejora de plan sin costo adicional ($120 fijo), y Contacto proactivo ($5). La utilidad esperada de cada acción se calcula como:

```
U(a) = p_churn × efectividad(a) × valor_cliente − costo(a)
```

---

## Notebook

El notebook [`Proyecto_Churn_Notebook.ipynb`](notebooks/Proyecto_Churn_Notebook.ipynb) contiene el flujo completo del proyecto:

1. **EDA** — Análisis exploratorio, distribución de variables, correlaciones y detección de valores atípicos
2. **Preprocesamiento** — Codificación binaria, one-hot encoding con drop_first, división train/test (80/20)
3. **Modelos baseline** — Regresión Logística, Random Forest y XGBoost evaluados con Accuracy
4. **Optimización con F2-score** — GridSearchCV sobre cada modelo usando F2-score como métrica objetivo (pondera Recall el doble que Precision)
5. **Evaluación final** — Comparativa de los 3 modelos optimizados en test (Accuracy, Precision, Recall, F1-score, F2-score, ROC-AUC)
6. **Exportación** — Serialización del modelo XGBoost final a `modelo_xgb_final.joblib` y `columnas_modelo.joblib`

---

## Código del sistema

```
├── deploy/                     # ★ Despliegue en Vercel
│   ├── api/
│   │   ├── main.py             # API FastAPI (/api/predecir y /api/recomendar)
│   │   ├── requirements.txt    # Dependencias de producción
│   │   └── modelo/             # Artefactos del modelo entrenado
│   │       ├── modelo_xgb_final.joblib
│   │       └── columnas_modelo.joblib
│   └── vercel.json             # Configuración de Vercel
├── notebooks/
│   └── Proyecto_Churn_Notebook.ipynb   # Notebook completo (EDA, entrenamiento, evaluación)
├── tests/
│   └── test_api.py             # Pruebas automatizadas (pytest)
├── docs/
│   └── Informe_Proyecto_Churn.docx
├── .gitignore
├── README.md
└── requirements-dev.txt
```

El **frontend** (Next.js, TypeScript, Tailwind) está en un repositorio separado: [alehpr22-source/churn-frontend](https://github.com/alehpr22-source/churn-frontend).

---

## Ejecución local

### API

```bash
cd deploy/api
pip install -r requirements.txt
uvicorn main:app --reload
```

Disponible en `http://localhost:8000/api`. Documentación Swagger en `http://localhost:8000/docs`.

### Pruebas

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Verifican que la API responda, que la probabilidad esté en [0, 1], que la acción recomendada corresponda a la mayor utilidad esperada y que clientes de bajo riesgo reciban "Sin accion".

### Frontend

Ver instrucciones en [churn-frontend](https://github.com/alehpr22-source/churn-frontend). Requiere `NEXT_PUBLIC_API_URL` apuntando a la API desplegada.

---

## Despliegue

### API (este repositorio)

1. En [vercel.com](https://vercel.com), importar este repositorio.
2. **Project Settings → Root Directory** → `deploy/`
3. Vercel detecta `vercel.json` automáticamente.

### Frontend

1. Importar [churn-frontend](https://github.com/alehpr22-source/churn-frontend) en Vercel (se detecta Next.js automáticamente).
2. Agregar variable de entorno: `NEXT_PUBLIC_API_URL = https://churn-api-alpha-five.vercel.app`
3. URL del frontend desplegado: [churn-prediction-frontend-one.vercel.app](https://churn-prediction-frontend-one.vercel.app)

**Nota:** Al ser funciones serverless, la primera consulta tras inactividad puede demorar unos segundos (cold start).
