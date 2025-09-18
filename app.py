import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# =========================================
# Configuración de la App
# =========================================
st.set_page_config(page_title="Previo ML - UIS", layout="wide")
st.title("Previo de Machine Learning - Primera Evaluación")
st.markdown("**Henry Lamos - Universidad Industrial de Santander - Ingeniería Industrial**")

# =========================================
# Sidebar - navegación
# =========================================
menu = st.sidebar.radio(
    "Módulos",
    ("Carga y Análisis de Datos", "Construcción y Evaluación de Modelos")
)

# Variables globales
df = None

# =========================================
# Módulo: Carga y Análisis de Datos
# =========================================
if menu == "Carga y Análisis de Datos":
    st.header("📂 Carga y Análisis de Datos")

    uploaded_file = st.file_uploader("Sube tu dataset en formato CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Vista previa del Dataset")
        st.dataframe(df.head(10))

        st.subheader("Estadísticas Descriptivas")
        st.write(df.describe())

        st.subheader("Información del Dataset")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Visualizaciones")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            st.write("Histogramas de variables numéricas")
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(6, 3 * len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col], kde=True, ax=axes[i])
                axes[i].set_title(col)
            st.pyplot(fig)

        if cat_cols:
            st.write("Gráficos de barras de variables categóricas")
            fig, axes = plt.subplots(len(cat_cols), 1, figsize=(6, 3 * len(cat_cols)))
            if len(cat_cols) == 1:
                axes = [axes]
            for i, col in enumerate(cat_cols):
                df[col].value_counts().plot(kind="bar", ax=axes[i])
                axes[i].set_title(col)
            st.pyplot(fig)

        if numeric_cols:
            st.write("Matriz de Correlación")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

# =========================================
# Módulo: Construcción y Evaluación de Modelos
# =========================================
elif menu == "Construcción y Evaluación de Modelos":
    st.header("⚙️ Construcción y Evaluación de Modelos")

    uploaded_file = st.file_uploader("Sube tu dataset en formato CSV", type=["csv"], key="model")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("Selección de Variables")
        target = st.selectbox("Selecciona la variable objetivo", df.columns)
        features = st.multiselect("Selecciona las variables predictoras", [col for col in df.columns if col != target])

        if target and features:
            X = df[features]
            y = df[target]

            # Codificación si es categórico
            if y.dtype == 'object':
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model_choice = st.selectbox("Selecciona el Modelo", 
                                        ["Árbol de Decisión", "Bagging", "Random Forest", "AdaBoost", "Gradient Boosting"])

            if model_choice == "Árbol de Decisión":
                criterion = st.radio("Criterio de División", ["gini", "entropy"])
                max_depth = st.slider("Profundidad Máxima", 1, 20, 3)
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)

            elif model_choice == "Bagging":
                base_estimator = DecisionTreeClassifier()
                n_estimators = st.slider("Número de Estimadores", 10, 200, 50)
                model = BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=42)

            elif model_choice == "Random Forest":
                n_estimators = st.slider("Número de Árboles", 10, 200, 100)
                max_depth = st.slider("Profundidad Máxima", 1, 20, 5)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            elif model_choice == "AdaBoost":
                n_estimators = st.slider("Número de Estimadores", 10, 200, 50)
                model = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)

            elif model_choice == "Gradient Boosting":
                n_estimators = st.slider("Número de Estimadores", 10, 200, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

            if st.button("Entrenar Modelo"):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.subheader("Matriz de Confusión")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

                st.subheader("Reporte de Clasificación")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                if len(np.unique(y)) == 2:
                    st.subheader("Curva ROC")
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax.plot([0, 1], [0, 1], "r--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)
