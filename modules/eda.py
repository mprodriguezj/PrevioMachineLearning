import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def perform_eda(df):
    
    # Pestañas para organizar el análisis
    tab1, tab2, tab3 = st.tabs([
        "Distribuciones", 
        "Correlaciones", 
        "Información General"
    ])
    
    # Obtener tipos de columnas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    with tab1:
        st.write("### Análisis de Distribuciones")
        
        # Selector de tipo de variable
        analysis_type = st.radio(
            "Tipo de análisis:",
            ["Variables Numéricas", "Variables Categóricas"],
            horizontal=True,
            key="dist_type"
        )
        
        if analysis_type == "Variables Numéricas" and numeric_cols:
            selected_numeric = st.selectbox(
                "Selecciona variable numérica:",
                numeric_cols,
                key="numeric_select"
            )
            
            if selected_numeric:
                col_data = df[selected_numeric].dropna()
                
                # Crear subplots para análisis completo
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma con Matplotlib
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(col_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_title(f'Histograma de {selected_numeric}')
                    ax.set_xlabel(selected_numeric)
                    ax.set_ylabel('Frecuencia')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # Box plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.boxplot(col_data, vert=True)
                    ax.set_title(f'Box Plot de {selected_numeric}')
                    ax.set_ylabel(selected_numeric)
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with col2:
                    # Gráfico de densidad
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.kdeplot(col_data, ax=ax, fill=True, color='orange', alpha=0.7)
                    ax.set_title(f'Densidad de {selected_numeric}')
                    ax.set_xlabel(selected_numeric)
                    ax.set_ylabel('Densidad')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    
                    # QQ Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    stats.probplot(col_data, dist="norm", plot=ax)
                    ax.set_title(f'QQ Plot de {selected_numeric}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Estadísticas de la distribución
                st.write("**Estadísticas Descriptivas:**")
                stats_data = {
                    'Estadística': ['Media', 'Mediana', 'Desviación Estándar', 'Mínimo', 'Máximo', 
                                  'Asimetría', 'Curtosis', 'Valores Nulos'],
                    'Valor': [
                        f"{col_data.mean():.2f}",
                        f"{col_data.median():.2f}",
                        f"{col_data.std():.2f}",
                        f"{col_data.min():.2f}",
                        f"{col_data.max():.2f}",
                        f"{col_data.skew():.2f}",
                        f"{col_data.kurtosis():.2f}",
                        f"{df[selected_numeric].isnull().sum()}"
                    ]
                }
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Detección de outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
                st.info(f"ℹ️ Se detectaron {len(outliers)} valores atípicos en {selected_numeric}")
                
        elif analysis_type == "Variables Categóricas" and categorical_cols:
            selected_categorical = st.selectbox(
                "Selecciona variable categórica:",
                categorical_cols,
                key="categorical_select"
            )
            
            if selected_categorical:
                value_counts = df[selected_categorical].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de barras
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.7)
                    ax.set_title(f'Distribución de {selected_categorical}')
                    ax.set_xlabel(selected_categorical)
                    ax.set_ylabel('Frecuencia')
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Añadir valores en las barras
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                with col2:
                    # Pie chart para categorías limitadas
                    if len(value_counts) <= 10:
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                              startangle=90, colors=plt.cm.Set3(np.arange(len(value_counts))))
                        ax.set_title(f'Proporción de {selected_categorical}')
                        st.pyplot(fig)
                    else:
                        st.info("ℹ️ Demasiadas categorías para gráfico de torta")
                        # Mostrar top 10 categorías
                        top_10 = value_counts.head(10)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(range(len(top_10)), top_10.values, color='lightgreen', alpha=0.7)
                        ax.set_title(f'Top 10 Categorías de {selected_categorical}')
                        ax.set_xlabel('Categorías')
                        ax.set_ylabel('Frecuencia')
                        ax.set_xticks(range(len(top_10)))
                        ax.set_xticklabels(top_10.index, rotation=45, ha='right')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                
                # Tabla de frecuencias
                st.write("**Tabla de Frecuencias:**")
                freq_df = pd.DataFrame({
                    'Categoría': value_counts.index,
                    'Frecuencia': value_counts.values,
                    'Porcentaje': (value_counts.values / len(df)) * 100
                })
                st.dataframe(
                    freq_df.style.format({'Porcentaje': '{:.1f}%'}), 
                    use_container_width=True,
                    height=400
                )
        
        else:
            st.warning("⚠️ No hay variables disponibles para el tipo de análisis seleccionado")
    
    with tab2:
        st.write("### Análisis de Correlaciones")
        
        if len(numeric_cols) > 1:
            # Selector de método de correlación
            corr_method = st.selectbox(
                "Método de correlación:", 
                ['pearson', 'spearman', 'kendall'],
                key="corr_method"
            )
            
            # Matriz de correlación
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            
            # Heatmap de correlación con Seaborn
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0, 
                square=True,
                ax=ax,
                fmt='.2f',
                cbar_kws={'shrink': 0.8}
            )
            ax.set_title(f'Matriz de Correlación ({corr_method.upper()})', fontsize=16, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)
            
            # Correlaciones fuertes
            st.write("**Correlaciones Fuertes (|r| > 0.7):**")
            strong_corr = corr_matrix.unstack().reset_index()
            strong_corr.columns = ['Variable1', 'Variable2', 'Correlación']
            strong_corr = strong_corr[
                (strong_corr['Variable1'] != strong_corr['Variable2']) & 
                (abs(strong_corr['Correlación']) > 0.7)
            ].sort_values('Correlación', ascending=False)
            
            if len(strong_corr) > 0:
                st.dataframe(
                    strong_corr.style.format({'Correlación': '{:.3f}'}), 
                    use_container_width=True
                )
                
                # Gráficos de dispersión para correlaciones fuertes
                st.write("** Gráficos de Dispersión para Correlaciones Fuertes:**")
                pairs_to_plot = strong_corr.head(4)
                
                for i, (_, row) in enumerate(pairs_to_plot.iterrows()):
                    fig = px.scatter(
                        df, 
                        x=row['Variable1'], 
                        y=row['Variable2'],
                        title=f"{row['Variable1']} vs {row['Variable2']} (r={row['Correlación']:.3f})",
                        trendline='ols'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ No hay correlaciones fuertes (|r| > 0.7)")
        else:
            st.warning("⚠️ Se necesitan al menos 2 variables numéricas para análisis de correlación")
    
    with tab3:
        st.write("### Información General del Dataset")
        
        # Resumen rápido
        st.write("**Resumen Rápido:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Filas", len(df))
        with col2:
            st.metric("Total Columnas", len(df.columns))
        with col3:
            st.metric("Columnas Numéricas", len(numeric_cols))
        with col4:
            st.metric("Columnas Categóricas", len(categorical_cols))
        
        # Información de tipos de datos
        st.write("**Tipos de Datos:**")
        dtype_df = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes.values,
            'No Nulos': df.notnull().sum().values,
            'Nulos': df.isnull().sum().values,
            '% Nulos': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(
            dtype_df.style.format({'% Nulos': '{:.1f}%'}), 
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Estadísticas descriptivas básicas
        if len(numeric_cols) > 0:
            st.write("**Estadísticas Descriptivas (Variables Numéricas):**")
            stats_df = df[numeric_cols].describe().T
            st.dataframe(stats_df.style.format('{:.2f}'), use_container_width=True)
        
        # Gráfico de valores nulos solo si hay nulos
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            st.write("**Valores Nulos por Columna:**")
            
            # Filtrar solo columnas con nulos para el gráfico
            columns_with_nulls = null_counts[null_counts > 0]
            null_percentage = (columns_with_nulls / len(df)) * 100
            
            missing_plot_df = pd.DataFrame({
                'Columna': columns_with_nulls.index,
                'Porcentaje Nulos': null_percentage
            }).sort_values('Porcentaje Nulos', ascending=False)
            
            fig, ax = plt.subplots(figsize=(max(10, len(columns_with_nulls) * 0.8), 6))
            bars = ax.bar(missing_plot_df['Columna'], missing_plot_df['Porcentaje Nulos'], 
                        color='red', alpha=0.7)
            ax.set_xlabel('Columnas')
            ax.set_ylabel('Porcentaje de Valores Nulos (%)')
            ax.set_title('Distribución de Valores Nulos por Columna')
            plt.xticks(rotation=45, ha='right')
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)