
# 🏦 Bank Marketing — Análisis Exploratorio de Datos

> **Caso de Estudio N°1 · Especialización en Python for Analytics**

## 📌 Descripción del Proyecto

Aplicación interactiva construida con **Streamlit** para el Análisis Exploratorio de Datos (EDA) 
del dataset **BankMarketing.csv**, correspondiente a una institución financiera cuya efectividad 
de campañas cayó del 12% al 8% en 6 meses.
El objetivo es descubrir **patrones y relaciones relevantes** entre las variables para apoyar 
la toma de decisiones estratégicas — sin construcción de modelos predictivos.

---
## 👤 Autor

| Campo | Detalle |
|-------|---------|
| **Nombre** | MIGUEL ANGEL JIMENEZ HAUAMANI |
| **Curso** | Especialización en Python for Analytics |
---

## 🎯 Módulos de la Aplicación

| Módulo | Contenido |
|--------|-----------|
| 🏠 **Home** | Presentación del proyecto, autor y dataset |
| 📂 **Cargar Dataset** | Upload del CSV, vista previa y dimensiones |
| 🔍 **Análisis EDA** | 10 ítems de análisis exploratorio completo |
| 📊 **Conclusiones** | 5 conclusiones estratégicas y recomendaciones |

### 10 Ítems de Análisis EDA

1. 📌 Información general (tipos de datos, nulos, dimensiones)
2. 🔤 Clasificación de variables (numéricas vs categóricas)
3. 📊 Estadísticas descriptivas (media, mediana, moda, dispersión)
4. ❓ Análisis de valores faltantes y 'unknown'
5. 📈 Distribución de variables numéricas (histogramas + KDE)
6. 📊 Distribución de variables categóricas (barras + proporciones)
7. 🔗 Análisis bivariado: numérico vs categórico (boxplot + violin)
8. 🔗 Análisis bivariado: categórico vs categórico (tabla de contingencia)
9. 🎛️ Análisis dinámico con filtros interactivos
10. 🏆 Hallazgos clave + Matriz de correlación

---

## 🛠️ Tecnologías Utilizadas

- 🐍 **Python 3.11**
- 📊 **Pandas & NumPy** — Manipulación de datos
- 📈 **Matplotlib & Seaborn** — Visualización
- 🌐 **Streamlit** — Interfaz web interactiva
- 🏗️ **POO (DataAnalyzer)** — Clase personalizada que encapsula el análisis

---

## 🚀 Instrucciones de Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/Miguel23JH/bank-marketing-eda.git
cd bank-marketing-eda
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app se abrirá automáticamente en tu navegador en `http://localhost:8501`.

### 5. Usar la aplicación

1. Ve al módulo **📂 Cargar Dataset**
2. Sube el archivo `BankMarketing.csv`
3. Navega por los módulos del sidebar

---

## 📁 Estructura del Repositorio

```
bank-marketing-eda/
│
├── app.py               # Aplicación principal Streamlit
├── BankMarketing.csv    # Dataset
├── requirements.txt     # Dependencias Python
└── README.md            # Este archivo
```

---

## 🔗 Links

- 🌐 App desplegada: https://github.com/Miguel23JH/marketing-bancario-eda
- 📁 Repositorio: https://bank-marketing-eda-njwsaoxzksrb2a4xgvekfn.streamlit.app/

---

## 📊 Principales Hallazgos

1. Los clientes con **éxito en campañas anteriores** tienen una tasa de conversión 6× mayor
2. La **duración de la llamada** es el predictor conductual más relevante
3. Los **segmentos etarios extremos** (<30 y >60 años) son más receptivos
4. El **contexto macroeconómico** (Euribor bajo) favorece la conversión
5. El **canal celular** supera consistentemente al teléfono fijo


