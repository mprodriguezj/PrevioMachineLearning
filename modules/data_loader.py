import streamlit as st
import pandas as pd
import numpy as np

def load_data():
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV para analizar", 
        type=['csv'],
        help="Formatos soportados: CSV",
        key="file_uploader_loader"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset cargado exitosamente: {df.shape[0]} filas × {df.shape[1]} columnas")
            return df
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")
            return None
    return None

def display_dataset_info(df):
    """
    Muestra solo información básica del dataset, sin duplicar el EDA completo
    """
    st.subheader("Vista Rápida del Dataset")
    
    # Pestañas básicas
    tab1, tab2 = st.tabs(["Vista Previa", "Información Básica"])
    
    with tab1:
        st.write("**Muestra del Dataset:**")
        
        # Selector de vista
        view_option = st.radio(
            "Tipo de vista:",
            ["Primeras 10 filas", "Últimas 5 filas", "Muestra aleatoria (10 filas)"],
            horizontal=True,
            key="view_option"
        )
        
        if view_option == "Primeras 10 filas":
            st.dataframe(df.head(10), use_container_width=True, height=300)
        elif view_option == "Últimas 5 filas":
            st.dataframe(df.tail(5), use_container_width=True, height=250)
        else:
            st.dataframe(df.sample(10, random_state=42), use_container_width=True, height=300)
    
    with tab2:
        st.write("**Información Básica:**")
        
        # Métricas en columnas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Filas", len(df))
        with col2:
            st.metric("Total Columnas", len(df.columns))
        with col3:
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numéricas", numeric_count)
        with col4:
            categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
            st.metric("Categóricas", categorical_count)
        
        # Información adicional
        st.write("**Información de Almacenamiento:**")
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.write(f"- Memoria usada: **{memory_usage:.2f} MB**")
        
        total_nulls = df.isnull().sum().sum()
        null_percentage = (total_nulls / (len(df) * len(df.columns))) * 100
        st.write(f"- Valores nulos totales: **{total_nulls}** ({null_percentage:.1f}%)")
        
        # Tipos de datos
        st.write("**Distribución de Tipos de Datos:**")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"- `{dtype}`: {count} columnas")