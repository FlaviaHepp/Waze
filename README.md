# 🗺️Predicción de abandono de usuarios en Waze (Churn Analysis)

Este proyecto aborda un **problema de machine learning supervisado**: la predicción de abandono de usuarios (*churn*) en la aplicación Waze, a partir de patrones de uso y comportamiento.

El objetivo es **entender qué variables influyen en la deserción** y preparar un dataset adecuado para el entrenamiento de modelos predictivos de clasificación.

---

## 🧠 Contexto del problema

La retención de usuarios es un factor clave en aplicaciones basadas en comunidad como Waze.  
Detectar tempranamente usuarios con alta probabilidad de abandono permite diseñar **estrategias de retención proactivas**.

Este proyecto se enfoca en:
- Comprender el comportamiento de usuarios retenidos vs. desertores
- Preparar variables relevantes para modelos de ML
- Sentar las bases para un pipeline de predicción de churn

---

## 🎯 Objetivo de machine learning

- **Tipo de problema:** Clasificación binaria
- **Variable objetivo:** `label`  
  - 0 → usuario retenido  
  - 1 → usuario que abandona
- **Resultado esperado:** modelo capaz de estimar la probabilidad de abandono de un usuario

---

## ❓ Preguntas que guía el análisis

- ¿Qué patrones de uso están asociados a una mayor probabilidad de churn?
- ¿La frecuencia de uso es más relevante que la intensidad (km, duración)?
- ¿Existen diferencias significativas por tipo de dispositivo?
- ¿Qué variables aportan mayor información predictiva?

---

## 📊 Dataset

- Datos de uso de la app Waze a nivel usuario
- Variables numéricas y categóricas relacionadas con:
  - Sesiones
  - Días de conducción
  - Kilómetros recorridos
  - Duración de los viajes
  - Antigüedad del usuario
- Dataset preparado para tareas de **clasificación supervisada**

---

## 🧪 Metodología

1. **Exploración de datos (EDA)**
   - Distribuciones, outliers y relaciones entre variables
2. **Feature Engineering**
   - Creación de variables como:
     - km por día de conducción
     - porcentaje de sesiones en el último mes
3. **Tratamiento de outliers**
   - Imputación por percentil 95
   - Eliminación basada en IQR
4. **Preparación para ML**
   - Codificación de variables categóricas
   - Eliminación de multicolinealidad
   - Dataset limpio y listo para modelado

---

## ⚙️ Técnicas de Machine Learning aplicadas

- Feature engineering
- Detección y tratamiento de valores atípicos
- Análisis de correlación y multicolinealidad
- Codificación de variables categóricas
- Preparación de datos para modelos supervisados
- Análisis de churn (clasificación binaria)

---

## 🔍 Principales insights

- La tasa de abandono es aproximadamente **17%**
- No se observan diferencias significativas de churn entre dispositivos
- Usuarios que recorren **grandes distancias en pocos días** presentan mayor probabilidad de abandono
- La **frecuencia de uso** está negativamente correlacionada con el churn
- Existe alta correlación entre `activity_days` y `driving_days`, por lo que se eliminó una de ellas

---

## 🛠️ Tecnologías utilizadas

- **Python**
- **pandas, numpy** → manipulación de datos
- **matplotlib, seaborn** → visualización
- **scikit-learn** → preprocessing y preparación para ML

---

## 📂 Estructura del repositorio

├── waze_app_dataset.csv
├── Uso de Waze.py
├── README.md


---

## 🚀 Próximos pasos (enfoque Data Scientist)

- Entrenamiento de modelos:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Evaluación con métricas:
  - ROC-AUC
  - Precision / Recall
  - F1-score
- Feature importance y explainability (SHAP)
- Optimización de hiperparámetros
- Simulación de estrategias de retención basadas en predicciones

---


**Flavia Hepp**  
Data Scientist en formación  
