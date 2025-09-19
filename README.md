# 🧠 Aplicación de Evaluación de Modelos de Machine Learning

## Descripción

Este proyecto es un taller desarrollado en el marco de la asignatura **Aprendizaje Automático** de la **Escuela de Estudios Industriales y Empresariales** de la **Universidad Industrial de Santander**.

Tiene como objetivo la **implementación, evaluación y despliegue** de modelos de aprendizaje automático, haciendo énfasis en clasificadores como **árboles de decisión** y **métodos de ensamble**.

La aplicación fue construida con **Streamlit** y permite una interacción visual e intuitiva con los modelos, desde la carga de datos hasta la interpretación de métricas.

---

## Características

- 📊 Análisis exploratorio de datos  
- 🌳 Árboles de decisión con diferentes criterios de división  
- 🤝 Modelos de ensamble: Random Forest, AdaBoost, Gradient Boosting, Bagging  
- 📈 Visualización de métricas de evaluación: matriz de confusión, clasificación, ROC y AUC  

---

## Instalación

Asegúrate de tener Python 3.10 o superior instalado (probado hasta 3.13.2). Luego, instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación, simplemente corre el siguiente comando desde la raíz del proyecto:

```bash
streamlit run app.py
```

Esto abrirá una interfaz gráfica en tu navegador donde podrás interactuar con los modelos disponibles.

## Estructura del Proyecto
.
├── app.py                       # Archivo principal que ejecuta la aplicación Streamlit
├── requirements.txt             # Archivo con las dependencias necesarias
├── data/                        # Carpeta para almacenar datasets utilizados en la aplicación
│ └── dataset.csv                # Dataset de ejemplo
├── modules/                     # Paquete con módulos funcionales de la aplicación
│ ├── data_loader.py             # Funciones para carga y validación de datos
│ ├── eda.py                     # Funciones para análisis exploratorio de datos (EDA)
│ ├── decision_tree.py           # Entrenamiento y evaluación de árboles de decisión
│ └── ensemble_models.py         # Implementación y evaluación de modelos de ensamble
└── README.md                    # Documentación general del proyecto