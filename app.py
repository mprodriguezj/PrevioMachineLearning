import streamlit as st
import pandas as pd
from modules.data_loader import load_data, display_dataset_info
from modules.eda import perform_eda
from modules.decision_tree import decision_tree_module
from modules.ensemble_models import ensemble_models_module

def main():
    st.set_page_config(
        page_title="ML Evaluation App - UIS",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Aplicación de Evaluación de Modelos de Machine Learning")
    st.markdown("""
    **Universidad Industrial de Santander**<br>
    **Materia:** Aprendizaje Automático - **Profesor:** Henry Lamos<br>
    **Desarrollado por:** Paula Rodríguez y Kevin Vera
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Selecciona un módulo:",
        ["Carga de Datos", "Análisis Exploratorio", "Árbol de Decisión", "Modelos de Ensamble"]
    )
    
    # Load data (available across all pages)
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    # Page routing
    if page == "Carga de Datos":
        data_loader_page()
    elif page == "Análisis Exploratorio" and st.session_state.df is not None:
        eda_page()
    elif page == "Árbol de Decisión" and st.session_state.df is not None:
        decision_tree_page()
    elif page == "Modelos de Ensamble" and st.session_state.df is not None:
        ensemble_models_page()
    elif st.session_state.df is None:
        st.warning("⚠️ Por favor carga un dataset primero en la pestaña 'Carga de Datos'")

def data_loader_page():
    st.header("📁 Carga y Análisis de Datos")
    df = load_data()
    if df is not None:
        st.session_state.df = df
        display_dataset_info(df)

def eda_page():
    st.header("🔍 Análisis Exploratorio de Datos")
    perform_eda(st.session_state.df)

def decision_tree_page():
    st.header("🌳 Árbol de Decisión")
    decision_tree_module(st.session_state.df)

def ensemble_models_page():
    st.header("🤝 Modelos de Ensamble")
    ensemble_models_module(st.session_state.df)

if __name__ == "__main__":
    main()