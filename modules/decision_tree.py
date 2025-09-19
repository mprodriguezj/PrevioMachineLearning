import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

def decision_tree_module(df):
    st.subheader("Configuración del Árbol de Decisión")
    
    # Data preparation
    st.write("**Preparación de datos:**")
    
    # Select target variable
    target_col = st.selectbox(
        "Selecciona la variable objetivo:",
        df.columns,
        key="target_select_dt"
    )
    
    # Select features
    available_features = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect(
        "Selecciona las variables predictoras:",
        available_features,
        default=available_features,
        key="features_select_dt"
    )
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    # Prepare data
    X = df[selected_features]
    y = df[target_col]
    
    # Handle categorical variables
    X = pd.get_dummies(X)
    
    # Split data
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state:", 0, 100, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Model configuration
    st.subheader("Hiperparámetros del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        criterion = st.selectbox(
            "Criterio de división:",
            ["gini", "entropy"],
            help="GINI: Medida de impureza | Entropy: Gain Ratio"
        )
        
        max_depth = st.slider("Profundidad máxima:", 1, 20, 5)
    
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2)
        min_samples_leaf = st.slider("Mínimo samples por hoja:", 1, 10, 1)
    
    # Train model
    if st.button("🚀 Entrenar Árbol de Decisión"):
        try:
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
            
            # Display results
            display_results(y_test, y_pred, y_prob, model.classes_)
            
        except Exception as e:
            st.error(f"❌ Error al entrenar el modelo: {str(e)}")

def display_results(y_test, y_pred, y_prob, classes):
    st.subheader("📊 Resultados de la Evaluación")
    
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curva ROC y AUC"])
    
    # --- Matriz de Confusión ---
    with tab1:
        st.write("**Matriz de Confusión:**")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    # --- Reporte de Clasificación Mejorado ---
    with tab2:
        st.subheader("📋 Reporte de Clasificación")
        
        # Calcular el reporte
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Separar métricas por clase y generales - FORMA CORRECTA
        # Las métricas globales están en claves específicas
        accuracy = report['accuracy']  # Exactitud global
        
        # Obtener promedios (pueden no existir si hay solo una clase)
        macro_avg = report.get('macro avg', {})
        weighted_avg = report.get('weighted avg', {})
        
        # Obtener métricas por clase (excluyendo las globales)
        class_metrics = {}
        for key in report.keys():
            if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(report[key], dict):
                class_metrics[key] = report[key]
        
        class_metrics_df = pd.DataFrame(class_metrics).transpose()
        
        # Mostrar métricas globales de manera visual y clara
        st.write("**Métricas Globales del Modelo**")
        
        # Crear columnas para las métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Exactitud (Accuracy)",
                value=f"{accuracy:.3f}",
                help="Porcentaje total de predicciones correctas"
            )
        
        with col2:
            precision_avg = weighted_avg.get('precision', 0) if weighted_avg else 0
            st.metric(
                label="Precisión Promedio",
                value=f"{precision_avg:.3f}",
                help="Capacidad del modelo para no predecir falsos positivos"
            )
        
        with col3:
            recall_avg = weighted_avg.get('recall', 0) if weighted_avg else 0
            st.metric(
                label="Recall Promedio", 
                value=f"{recall_avg:.3f}",
                help="Capacidad del modelo para encontrar todos los positivos"
            )
        
        with col4:
            f1_avg = weighted_avg.get('f1-score', 0) if weighted_avg else 0
            st.metric(
                label="F1-Score Promedio",
                value=f"{f1_avg:.3f}",
                help="Balance entre Precisión y Recall"
            )
        
        # Explicación de las métricas
        with st.expander("📚 ¿Qué significan estas métricas?"):
            st.markdown("""
            - **Exactitud (Accuracy)**: Porcentaje de predicciones correctas sobre el total.
            - **Precisión (Precision)**: De todos los que predije como positivos, ¿cuántos realmente lo eran?
            - **Recall (Sensibilidad)**: De todos los reales positivos, ¿cuántos logré identificar?
            - **F1-Score**: Media armónica entre Precisión y Recall (balance entre ambas).
            - **Soporte (Support)**: Número de muestras reales de cada clase.
            """)
        
        # Mostrar tabla con métricas por clase si hay múltiples clases
        if not class_metrics_df.empty:
            st.write("**Métricas por Clase**")
            
            # Formatear el dataframe para mejor visualización
            class_metrics_display = class_metrics_df.copy()
            class_metrics_display.index.name = 'Clase'
            class_metrics_display = class_metrics_display.reset_index()
            
            # Mostrar tabla con métricas por clase
            st.dataframe(
                class_metrics_display.style.format({
                    'precision': '{:.3f}',
                    'recall': '{:.3f}', 
                    'f1-score': '{:.3f}',
                    'support': '{:.0f}'
                }).highlight_max(subset=['precision', 'recall', 'f1-score'], color='#90EE90')
                .highlight_min(subset=['precision', 'recall', 'f1-score'], color='#FFCCCB'),
                use_container_width=True,
                height=min(400, 150 + len(class_metrics_df) * 35)
            )
        
        # Mostrar comparación entre promedios solo si existen
        if macro_avg and weighted_avg:
            st.write("**Comparación de Promedios**")
            
            # Crear gráfico de comparación simple
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ['Precisión', 'Recall', 'F1-Score']
            macro_values = [
                macro_avg.get('precision', 0),
                macro_avg.get('recall', 0), 
                macro_avg.get('f1-score', 0)
            ]
            weighted_values = [
                weighted_avg.get('precision', 0),
                weighted_avg.get('recall', 0),
                weighted_avg.get('f1-score', 0)
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, macro_values, width, label='Macro Promedio', alpha=0.8, color='#FF9999')
            bars2 = ax.bar(x + width/2, weighted_values, width, label='Promedio Ponderado', alpha=0.8, color='#66B2FF')
            
            ax.set_ylabel('Valor')
            ax.set_title('Comparación entre Macro Promedio y Promedio Ponderado')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim(0, 1.05)
            
            # Añadir valores en las barras
            for bar, value in zip(bars1, macro_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            for bar, value in zip(bars2, weighted_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Explicación de la diferencia entre promedios
            ax.text(0.02, 0.98, 
                    "Macro: Promedia sin considerar desbalance de clases\nPonderado: Considera el tamaño de cada clase",
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("ℹ️ Las métricas de promedio solo están disponibles para clasificación multiclase")
    
    # --- Curva ROC y AUC ---
    with tab3:
        st.write("### Curva ROC (Receiver Operating Characteristic)")
        st.write("""
        La **Curva ROC** es una representación gráfica que muestra la capacidad de un clasificador 
        para diferenciar entre clases. Se basa en dos métricas:
        
        - **Tasa de falsos positivos (False Positive Rate - FPR):** Proporción de negativos incorrectamente clasificados como positivos.
        - **Tasa de verdaderos positivos (True Positive Rate - TPR):** Proporción de positivos correctamente identificados.
        
        La curva muestra la relación entre TPR y FPR para diferentes umbrales de decisión.
        """)
        
        st.write("### Área bajo la curva (AUC)")
        st.write("""
        El **AUC** cuantifica la calidad de la curva ROC en un solo valor:
        
        - **0.9 - 1.0:** Excelente poder discriminativo
        - **0.8 - 0.9:** Muy bueno
        - **0.7 - 0.8:** Aceptable
        - **0.6 - 0.7:** Pobre
        - **0.5 - 0.6:** No mejor que aleatorio
        - **< 0.5:** Peor que aleatorio
        """)
        
        n_classes = len(classes)
        
        try:
            # Verificar y alinear las clases entre y_test y las clases del modelo
            unique_test_classes = np.unique(y_test)
            
            # Si hay clases en test que no están en el entrenamiento
            if len(unique_test_classes) != n_classes:
                st.warning(f"⚠️ Advertencia: Hay {len(unique_test_classes)} clases en test pero {n_classes} en entrenamiento")
                
                # Filtrar solo las muestras que pertenecen a las clases conocidas
                mask = y_test.isin(classes) if hasattr(y_test, 'isin') else np.isin(y_test, classes)
                y_test_filtered = y_test[mask]
                y_pred_filtered = y_pred[mask]
                y_prob_filtered = y_prob[mask]
                
                if len(y_test_filtered) == 0:
                    st.error("❌ No hay muestras en test que coincidan con las clases de entrenamiento")
                    return
                
                st.info(f"✅ Usando {len(y_test_filtered)} muestras de test que coinciden con las clases de entrenamiento")
                
                y_test = y_test_filtered
                y_pred = y_pred_filtered
                y_prob = y_prob_filtered
            
            y_test_bin = label_binarize(y_test, classes=classes)
            
            # --- GRÁFICO 1: CURVA ROC ---
            st.subheader("📈 Curva ROC")

            # Ajustar tamaño de figura proporcionalmente
            fig_width = 12  # Aumentar ancho para acomodar la leyenda
            fig_height = 8  # Mantener altura adecuada
            fig_roc, ax_roc = plt.subplots(figsize=(fig_width, fig_height))

            if n_classes == 2:
                # Clasificación binaria
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                ax_roc.plot(fpr, tpr, color='darkorange', lw=3, label=f'Curva ROC (AUC = {roc_auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea base (AUC = 0.5)')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
                ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
                ax_roc.set_title('Curva ROC - Clasificación Binaria', fontsize=13, fontweight='bold')
                
                # Leyenda fuera del gráfico con tamaño de fuente reducido
                ax_roc.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
                ax_roc.grid(True, alpha=0.3)
                
                # Ajustar tamaño de ticks
                ax_roc.tick_params(axis='both', which='major', labelsize=10)
                
            else:
                # Clasificación multiclase
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                colors = sns.color_palette("husl", n_classes)
                
                # Calcular AUC para cada clase
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    ax_roc.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                            label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
                
                # Micro promedio
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                ax_roc.plot(fpr["micro"], tpr["micro"],
                        label=f'Micro-promedio (AUC = {roc_auc["micro"]:.3f})',
                        color='black', linestyle=':', linewidth=3)
                
                ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Línea base (AUC = 0.5)')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
                ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
                ax_roc.set_title('Curva ROC - Clasificación Multiclase', fontsize=13, fontweight='bold')
                
                # Leyenda fuera del gráfico con tamaño de fuente reducido
                ax_roc.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=9)
                ax_roc.grid(True, alpha=0.3)
                
                # Ajustar tamaño de ticks
                ax_roc.tick_params(axis='both', which='major', labelsize=10)

            # Ajustar el layout para hacer espacio para la leyenda
            plt.tight_layout(rect=[0, 0, 0.88, 1])  # Ajuste proporcional del espacio
            st.pyplot(fig_roc)
            
            # --- GRÁFICO 2: MÉTRICAS AUC ---
            st.subheader("📊 Métricas AUC")
            fig_auc, ax_auc = plt.subplots(figsize=(10, 6))
            
            if n_classes == 2:
                # Gráfico de métricas AUC para binario
                metrics_data = [roc_auc]
                metric_labels = ['AUC']
                colors = ['lightgreen' if roc_auc >= 0.7 else 'lightcoral']
                
                bars = ax_auc.bar(metric_labels, metrics_data, color=colors, edgecolor='black', alpha=0.8)
                ax_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Aleatorio')
                ax_auc.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Aceptable')
                ax_auc.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excelente')
                ax_auc.set_ylim(0, 1.1)
                ax_auc.set_ylabel('Valor AUC')
                ax_auc.set_title('Métrica AUC - Clasificación Binaria')
                ax_auc.legend()
                
                # Añadir valores en las barras
                for bar, v in zip(bars, metrics_data):
                    height = bar.get_height()
                    ax_auc.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                
            else:
                # Gráfico de barras para AUC por clase en multiclase
                auc_scores = [roc_auc[i] for i in range(n_classes)]
                colors = sns.color_palette("husl", n_classes)
                
                bars = ax_auc.bar(range(n_classes), auc_scores, color=colors, edgecolor='black', alpha=0.8)
                ax_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Aleatorio')
                ax_auc.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Aceptable')
                ax_auc.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excelente')
                ax_auc.set_xticks(range(n_classes))
                ax_auc.set_xticklabels(classes, rotation=45, ha='right')
                ax_auc.set_ylabel('Valor AUC')
                ax_auc.set_title('AUC por Clase - Clasificación Multiclase')
                ax_auc.set_ylim(0, 1.1)
                ax_auc.legend()
                
                # Añadir valores en las barras
                for bar, score in zip(bars, auc_scores):
                    height = bar.get_height()
                    ax_auc.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig_auc)
            
            # --- MOSTRAR MÉTRICAS NUMÉRICAS ---
            if n_classes == 2:
                st.subheader("📈 Métricas de Evaluación")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AUC Score", f"{roc_auc:.4f}")
                with col2:
                    st.metric("Interpretación", interpretar_auc(roc_auc))
                with col3:
                    quality = "✅ Excelente" if roc_auc >= 0.9 else "👍 Buena" if roc_auc >= 0.8 else "⚠️ Aceptable" if roc_auc >= 0.7 else "❌ Pobre"
                    st.metric("Calidad", quality)
            
            else:
                # Métricas promedio para multiclase
                macro_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
                micro_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='micro')
                weighted_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                
                st.subheader("📈 Métricas de Evaluación AUC")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Macro AUC", f"{macro_auc:.4f}", help="Promedio simple de AUC por clase")
                with col2:
                    st.metric("Micro AUC", f"{micro_auc:.4f}", help="Promedio ponderado por el tamaño de cada clase")
                with col3:
                    st.metric("Weighted AUC", f"{weighted_auc:.4f}", help="Promedio ponderado por soporte de clase")
                
                # Tabla detallada por clase
                st.subheader("📋 AUC por Clase - Detalle")
                auc_data = []
                for i, class_name in enumerate(classes):
                    auc_data.append({
                        'Clase': class_name,
                        'AUC': f"{roc_auc[i]:.4f}",
                        'Interpretación': interpretar_auc(roc_auc[i]),
                        'Muestras en Test': np.sum(y_test == class_name)
                    })
                
                auc_df = pd.DataFrame(auc_data)
                st.dataframe(auc_df, use_container_width=True, hide_index=True)
                
                # Interpretación general
                st.subheader("🎯 Interpretación General")
                interpretacion = interpretar_auc(macro_auc)
                st.write(f"**Macro AUC ({macro_auc:.3f}):** {interpretacion}")
                
                if macro_auc >= 0.9:
                    st.success("🌟 Excelente poder discriminativo general del modelo")
                elif macro_auc >= 0.8:
                    st.info("📊 Buen poder discriminativo general del modelo")
                elif macro_auc >= 0.7:
                    st.warning("⚠️ Poder discriminativo general aceptable")
                else:
                    st.error("❌ Poder discriminativo general pobre")
                    
        except Exception as e:
            st.error(f"❌ Error al calcular las curvas ROC: {str(e)}")
            st.info("Esto puede ocurrir cuando hay problemas con las probabilidades predichas o las clases objetivo")

def interpretar_auc(auc_score):
    """Función auxiliar para interpretar scores AUC"""
    if auc_score >= 0.9:
        return "Excelente discriminación"
    elif auc_score >= 0.8:
        return "Muy buena discriminación"
    elif auc_score >= 0.7:
        return "Discriminación aceptable"
    elif auc_score >= 0.6:
        return "Discriminación pobre"
    elif auc_score >= 0.5:
        return "No mejor que aleatorio"
    else:
        return "Peor que aleatorio"