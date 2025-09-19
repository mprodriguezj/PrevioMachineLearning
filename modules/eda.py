import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(df):
    st.subheader("Visualizaciones Exploratorias")
    
    # Select columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if numeric_cols:
            st.write("**Variables Numéricas:**")
            selected_numeric = st.selectbox(
                "Selecciona variable numérica:",
                numeric_cols,
                key="numeric_select"
            )
            
            if selected_numeric:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_numeric].hist(bins=30, ax=ax)
                ax.set_title(f'Distribución de {selected_numeric}')
                ax.set_xlabel(selected_numeric)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
    
    with col2:
        if categorical_cols:
            st.write("**Variables Categóricas:**")
            selected_categorical = st.selectbox(
                "Selecciona variable categórica:",
                categorical_cols,
                key="categorical_select"
            )
            
            if selected_categorical:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[selected_categorical].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Distribución de {selected_categorical}')
                ax.set_xlabel(selected_categorical)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(fig)
    
    # Correlation matrix
    if len(numeric_cols) > 1:
        st.subheader("Matriz de Correlación")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Matriz de Correlación')
        st.pyplot(fig)