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
    
    # BOTÓN SAMBAIR DATASET - Añadido en el sidebar
    st.sidebar.markdown("---")  # Separador
    if st.sidebar.button("🔄 Cambiar Dataset", help="Reiniciar y cargar un nuevo dataset"):
        change_dataset()
    
    # Show current dataset info in sidebar if loaded
    if st.session_state.data_loaded:
        st.sidebar.success("Dataset cargado")
        st.sidebar.write(f"Filas: {len(st.session_state.df)}")
        st.sidebar.write(f"Columnas: {len(st.session_state.df.columns)}")
    
    # Page routing
    if page == "Carga de Datos":
        data_loader_page()
    elif page == "Análisis Exploratorio":
        if st.session_state.data_loaded:
            eda_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero")
    elif page == "Árbol de Decisión":
        if st.session_state.data_loaded:
            decision_tree_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero")
    elif page == "Modelos de Ensamble":
        if st.session_state.data_loaded:
            ensemble_models_page()
        else:
            st.warning("⚠️ Por favor carga un dataset primero")

def change_dataset():
    """Función para reiniciar y cambiar el dataset"""
    # Reiniciar todo el estado de la sesión
    st.session_state.df = None
    st.session_state.data_loaded = False
    st.session_state.current_page = "Carga de Datos"
    
    # Limpiar cualquier caché o estado adicional que puedas tener
    keys_to_remove = []
    for key in st.session_state.keys():
        if key not in ['_pages', '_last_page', '_scriptrunner']:
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Forzar redirección a la página de carga de datos
    st.session_state.current_page = "Carga de Datos"
    st.rerun()

def generate_sample_dataset():
    """Función para generar un dataset de ejemplo aleatorio"""
    try:
        # Generar datos de ejemplo
        np.random.seed(42)
        n_samples = 200
        
        # Crear dataset con múltiples características
        data = {
            'edad': np.random.randint(18, 70, n_samples),
            'ingresos': np.random.randint(20000, 100000, n_samples),
            'score_credito': np.random.randint(300, 850, n_samples),
            'monto_prestamo': np.random.randint(5000, 50000, n_samples),
            'plazo_prestamo': np.random.randint(12, 60, n_samples),
            'historial_credito': np.random.choice(['bueno', 'regular', 'malo'], n_samples, p=[0.6, 0.3, 0.1]),
            'empleo': np.random.choice(['empleado', 'independiente', 'desempleado'], n_samples, p=[0.7, 0.2, 0.1]),
            'educacion': np.random.choice(['bachiller', 'universitario', 'posgrado'], n_samples, p=[0.4, 0.4, 0.2]),
            'aprobado': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # Variable objetivo
        }
        
        df = pd.DataFrame(data)
        st.session_state.df = df
        st.session_state.data_loaded = True
        
        st.sidebar.success("✅ Dataset de ejemplo generado exitosamente!")
        st.sidebar.write(f"📊 {n_samples} muestras generadas")
        st.sidebar.write(f"🏷️ Variable objetivo: 'aprobado'")
        
        # Mostrar información del dataset en la página principal
        if st.session_state.get('current_page') == 'Carga de Datos':
            st.success("🎲 Dataset de ejemplo 'Sambair' generado exitosamente!")
            display_dataset_info(df)
        
    except Exception as e:
        st.sidebar.error(f"❌ Error al generar dataset: {str(e)}")

def data_loader_page():
    st.header("📁 Carga de Datos")
    st.session_state.current_page = "Carga de Datos"
    df = load_data()
    if df is not None:
        st.session_state.df = df
        st.session_state.data_loaded = True
        display_dataset_info(df)
    elif st.session_state.data_loaded:
        # Mostrar el dataset actual si ya está cargado
        display_dataset_info(st.session_state.df)

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