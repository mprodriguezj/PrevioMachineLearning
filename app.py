import sys
import os

# Añadir la carpeta del proyecto al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    st.title("Aplicación de Evaluación de Modelos de Machine Learning")
    st.markdown("""
    **Universidad Industrial de Santander**  
    **Materia:** Aprendizaje Automático - **Profesor:** Henry Lamos  
    **Desarrollado por:** Paula Rodríguez y Kevin Vera
    """)
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Sidebar navigation
    st.sidebar.title("Navegación")
    
    # Opciones de página con emojis únicos
    page_options = ["Carga de Datos", "Análisis Exploratorio", "Árbol de Decisión", "Modelos de Ensamble"]
    page = st.sidebar.radio("Selecciona un módulo:", page_options)
    
    # BOTÓN PARA CARGAR NUEVO DATASET
    st.sidebar.markdown("---")  # Separador
    if st.sidebar.button("🔄 Cargar Nuevo Dataset", help="Reiniciar y cargar un nuevo dataset"):
        change_dataset()
    
    # Show current dataset info in sidebar if loaded
    if st.session_state.data_loaded:
        st.sidebar.success("Dataset cargado")
        st.sidebar.write(f"Filas: {len(st.session_state.df)}")
        st.sidebar.write(f"Columnas: {len(st.session_state.df.columns)}")
    else:
        st.sidebar.info("ℹ️ Sube un dataset para comenzar")
    
    # Page routing
    if page == "Carga de Datos":
        data_loader_page()
    elif page == "Análisis Exploratorio":
        if st.session_state.data_loaded:
            eda_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero en la pestaña 'Carga de Datos'")
    elif page == "Árbol de Decisión":
        if st.session_state.data_loaded:
            decision_tree_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero en la pestaña 'Carga de Datos'")
    elif page == "Modelos de Ensamble":
        if st.session_state.data_loaded:
            ensemble_models_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero en la pestaña 'Carga de Datos'")

def change_dataset():
    """Función para reiniciar y cargar un nuevo dataset"""
    # Reiniciar todo el estado de la sesión
    st.session_state.df = None
    st.session_state.data_loaded = False
    
    # Limpiar cualquier caché o estado adicional
    keys_to_remove = []
    for key in st.session_state.keys():
        if key not in ['_pages', '_last_page', '_scriptrunner']:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    st.success("✅ Listo para cargar un nuevo dataset")
    st.rerun()

def data_loader_page():
    st.header("📁 Carga de Datos")
    st.session_state.current_page = "Carga de Datos"
    
    if not st.session_state.data_loaded:
        st.info("📤 **Instrucciones:** Sube tu archivo de dataset (.CSV) para comenzar el análisis.")
        
        df = load_data()
        if df is not None:
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success("✅ Dataset cargado exitosamente!")
            display_dataset_info(df)
    else:
        # Mostrar el dataset actual si ya está cargado
        st.success("✅ Dataset cargado - Puedes cambiar al módulo de análisis")
        display_dataset_info(st.session_state.df)
        
        # Opción para cargar nuevo dataset desde esta página también
        if st.button("🔄 Cargar Nuevo Dataset desde aquí"):
            change_dataset()

def eda_page():
    st.header("🔍 Análisis Exploratorio de Datos")
    st.session_state.current_page = "Análisis Exploratorio"
    if st.session_state.data_loaded:
        perform_eda(st.session_state.df)

def decision_tree_page():
    st.header("🌳 Árbol de Decisión")
    st.session_state.current_page = "Árbol de Decisión"
    if st.session_state.data_loaded:
        decision_tree_module(st.session_state.df)

def ensemble_models_page():
    st.header("🧩 Modelos de Ensamble")
    st.session_state.current_page = "Modelos de Ensamble"
    if st.session_state.data_loaded:
        ensemble_models_module(st.session_state.df)

if __name__ == "__main__":
    main()