import streamlit as st
import pandas as pd

def load_data():
    st.subheader("Carga de Dataset")
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV", 
        type=['csv'],
        help="Selecciona un archivo CSV para analizar"
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
    st.subheader("Información del Dataset")
    
    tab1, tab2, tab3 = st.tabs(["Primeras Filas", "Estadísticas", "Info y Valores Nulos"])
    
    with tab1:
        st.write("**Primeras 10 filas:**")
        st.dataframe(df.head(10))
    
    with tab2:
        st.write("**Estadísticas descriptivas:**")
        st.dataframe(df.describe())
    
    with tab3:
        st.write("**Información del dataset:**")
        
        # Create info buffer
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        st.text(info_str)
        
        # Missing values
        st.write("**Valores nulos por columna:**")
        missing_df = pd.DataFrame({
            'Columna': df.columns,
            'Valores Nulos': df.isnull().sum(),
            'Porcentaje Nulos': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df)