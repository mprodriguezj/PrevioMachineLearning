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
    
    # --- Preparación de datos ---
    st.write("**Preparación de datos:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Selección de variable objetivo
        target_col = st.selectbox(
            "Selecciona la variable objetivo:",
            df.columns,
            key="target_select_dt"
        )
    
    with col2:
        # Selección de variables predictoras
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
    
    # Construir X, y
    X = df[selected_features]
    y = df[target_col]
    
    # --- Limpieza y procesamiento de datos ---
    st.write("**Limpieza y procesamiento de datos:**")
    
    # Limpieza de valores nulos
    missing_rows = X.isnull().any(axis=1) | y.isnull()
    if missing_rows.any():
        st.warning(f"⚠️ Se eliminaron {missing_rows.sum()} filas con valores nulos")
        X = X[~missing_rows]
        y = y[~missing_rows]
    
    # Manejo de variables categóricas
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        st.info(f"ℹ️ Variables categóricas detectadas: {', '.join(categorical_cols)}")
        try:
            X = pd.get_dummies(X, drop_first=True)
            st.success(f"✅ Variables convertidas a one-hot encoding. Nuevas dimensiones: {X.shape}")
        except Exception as e:
            st.error(f"❌ Error al convertir variables categóricas: {str(e)}")
            return
    else:
        st.info("ℹ️ No se detectaron variables categóricas en los predictores")
    
    # Procesamiento de variable objetivo
    y = y.astype(str).str.strip()
    y = y.replace("?", np.nan)
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    # Eliminación de clases raras
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index
    if len(rare_classes) > 0:
        st.warning(f"⚠️ Se eliminaron {len(rare_classes)} clases con menos de 2 muestras")
        mask = ~y.isin(rare_classes)
        X = X[mask]
        y = y[mask]
    
    # Validación final
    if len(y.unique()) < 2:
        st.error("❌No hay suficientes clases para entrenar el modelo. Se necesitan al menos 2 clases diferentes.")
        return
    
    # --- Configuración de división de datos ---
    st.write("**Configuración de división de datos:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random state:", 0, 100, 42)
    
    # Validar estratificación
    can_stratify = len(y.unique()) >= 2 and y.value_counts().min() >= 2
    
    # División del dataset
    try:
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
                shuffle=True
            )
            st.success("✅ División estratificada realizada correctamente")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                shuffle=True
            )
            st.info("ℹ️ División no estratificada realizada")
    except ValueError as e:
        st.warning(f"⚠️ División estratificada falló: {e}. Reintentando sin estratificación...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
    
    # --- Configuración de hiperparámetros ---
    st.subheader("Configuración de Hiperparámetros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        criterion = st.selectbox(
            "Criterio de división:",
            ["gini", "entropy"],
            help="GINI: medida de impureza | Entropy: ganancia de información"
        )
        max_depth = st.slider("Profundidad máxima:", 1, 20, 5)
    
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2)
        min_samples_leaf = st.slider("Mínimo samples por hoja:", 1, 10, 1)
    
    # --- Entrenamiento del modelo ---
    if st.button("Entrenar Árbol de Decisión", type="primary"):
        try:
            if len(np.unique(y_train)) < 2:
                st.error("❌ El conjunto de entrenamiento debe tener al menos 2 clases diferentes")
                return
            
            with st.spinner("Entrenando modelo..."):
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    y_prob = None
                
                st.success("✅ Modelo entrenado exitosamente")
                display_results(y_test, y_pred, y_prob, model.classes_)
        
        except Exception as e:
            st.error(f"❌ Error al entrenar el modelo: {str(e)}")
            st.write("Sugerencia: Verifica la variable objetivo, ajusta los hiperparámetros o revisa las variables predictoras.")

def display_results(y_test, y_pred, y_prob, classes):
    st.subheader("Resultados de la Evaluación")
    
    # Determinar el tipo de problema
    n_classes = len(classes)
    is_binary_classification = n_classes == 2
    is_multiclass_classification = n_classes > 2
    
    # Crear pestañas según el tipo de problema
    if is_binary_classification and y_prob is not None:
        tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curva ROC y AUC"])
    else:
        tab1, tab2 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación"])
    
    # Matriz de Confusión
    with tab1:
        st.write("**Matriz de Confusión:**")
        cm = confusion_matrix(y_test, y_pred)

        num_classes = len(classes)

        # 🔧 AJUSTES DINÁMICOS MEJORADOS para matriz de confusión
        if num_classes == 2:
            # Tamaño especial para 2 clases (como sex)
            fig_width, fig_height = 6, 5
            font_size = 16
            tick_font_size = 14
            annotation_size = 18
        elif num_classes <= 5:
            fig_width, fig_height = 8, 7
            font_size = 14
            tick_font_size = 12
            annotation_size = 14
        elif num_classes <= 10:
            fig_width = min(1 + num_classes * 0.5, 12)
            fig_height = min(1 + num_classes * 0.5, 12)
            font_size = 10
            tick_font_size = 10
            annotation_size = 10
        else:
            fig_width = min(1 + num_classes * 0.5, 20)
            fig_height = min(1 + num_classes * 0.5, 20)
            font_size = 8
            tick_font_size = 8
            annotation_size = 8

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Crear heatmap con mejoras visuales
        heatmap = sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={'size': annotation_size, 'weight': 'bold'},
            cbar_kws={'shrink': 0.7},
            square=True  # Hace la matriz cuadrada para mejor aspecto
        )

        # Mejorar etiquetas y título
        ax.set_xlabel('Predicciones', fontsize=tick_font_size + 2, weight='bold')
        ax.set_ylabel('Valores Reales', fontsize=tick_font_size + 2, weight='bold')
        ax.set_title('Matriz de Confusión', fontsize=font_size + 4, weight='bold', pad=20)

        # Rotar etiquetas solo si hay muchas clases
        if num_classes > 5:
            rotation = 45
            ha = 'right'
        else:
            rotation = 0
            ha = 'center'

        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha=ha, fontsize=tick_font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_font_size)

        # Añadir líneas de separación más visibles para pocas clases
        if num_classes <= 5:
            for i in range(num_classes + 1):
                ax.axhline(i, color='white', linewidth=2)
                ax.axvline(i, color='white', linewidth=2)

        plt.tight_layout()
        st.pyplot(fig)

        # Mostrar advertencia si hay demasiadas clases
        if num_classes > 15:
            st.warning("⚠️ Hay muchas clases en la variable objetivo. Considera agrupar clases similares para una mejor visualización.")

        # Información adicional para matrices pequeñas
        if num_classes == 2:
            st.info("""
            **📊 Interpretación para 2 clases:**
            - **Verdaderos Negativos (TN)**: Casos negativos correctamente clasificados
            - **Falsos Positivos (FP)**: Casos negativos incorrectamente clasificados como positivos
            - **Falsos Negativos (FN)**: Casos positivos incorrectamente clasificados como negativos  
            - **Verdaderos Positivos (TP)**: Casos positivos correctamente clasificados
            """)

    # Reporte de Clasificación Simplificado
    with tab2:
        st.subheader("Reporte de Clasificación")
        
        if y_test is not None and y_pred is not None:
            # Calcular el reporte
            report = classification_report(y_test, y_pred, output_dict=True)
            
            accuracy = report.get('accuracy', 0)
            macro_avg = report.get('macro avg', {})
            weighted_avg = report.get('weighted avg', {})
            
            # Obtener métricas por clase
            class_metrics = {}
            for key in report.keys():
                if key not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(report[key], dict):
                    class_metrics[key] = report[key]
            
            # Métricas Globales
            st.write("**Métricas Globales del Modelo**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Exactitud (Accuracy)",
                    value=f"{accuracy:.3f}",
                    help="Porcentaje total de predicciones correctas"
                )
            
            with col2:
                precision_avg = weighted_avg.get('precision', macro_avg.get('precision', 0))
                st.metric(
                    label="Precisión Promedio",
                    value=f"{precision_avg:.3f}",
                    help="Capacidad del modelo para no predecir falsos positivos"
                )
            
            with col3:
                recall_avg = weighted_avg.get('recall', macro_avg.get('recall', 0))
                st.metric(
                    label="Recall Promedio", 
                    value=f"{recall_avg:.3f}",
                    help="Capacidad del modelo para encontrar todos los positivos"
                )
            
            with col4:
                f1_avg = weighted_avg.get('f1-score', macro_avg.get('f1-score', 0))
                st.metric(
                    label="F1-Score Promedio",
                    value=f"{f1_avg:.3f}",
                    help="Balance entre Precisión y Recall"
                )

            # Mostrar métricas por clase si hay múltiples clases
            if class_metrics and len(class_metrics) > 1:
                class_metrics_df = pd.DataFrame(class_metrics).transpose()
                
                # Gráfico de rendimiento por clase
                st.write("**Rendimiento por Clase**")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                classes = class_metrics_df.index
                x = np.arange(len(classes))
                width = 0.25
                
                # Crear barras para cada métrica
                bars1 = ax.bar(x - width, class_metrics_df['precision'], width, label='Precisión', alpha=0.8, color='#FF6B6B')
                bars2 = ax.bar(x, class_metrics_df['recall'], width, label='Recall', alpha=0.8, color='#4ECDC4')
                bars3 = ax.bar(x + width, class_metrics_df['f1-score'], width, label='F1-Score', alpha=0.8, color='#45B7D1')
                
                # Personalizar gráfico
                ax.set_xlabel('Clases')
                ax.set_ylabel('Puntuación')
                ax.set_title('Métricas por Clase')
                ax.set_xticks(x)
                ax.set_xticklabels(classes, rotation=45, ha='right')
                ax.legend()
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tabla de métricas
                st.write("**Tabla de Métricas por Clase**")
                
                class_metrics_display = class_metrics_df.copy()
                class_metrics_display.index.name = 'Clase'
                class_metrics_display = class_metrics_display.reset_index()
                
                st.dataframe(
                    class_metrics_display.style.format({
                        'precision': '{:.3f}',
                        'recall': '{:.3f}', 
                        'f1-score': '{:.3f}',
                        'support': '{:.0f}'
                    }),
                    use_container_width=True,
                    height=min(300, 100 + len(class_metrics_df) * 35)
                )
            
            elif class_metrics and len(class_metrics) == 1:
                # Caso binario
                st.info("Clasificación binaria")
                class_name = list(class_metrics.keys())[0]
                metrics = class_metrics[class_name]
                
                st.write(f"**Métricas para la clase {class_name}:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precisión", f"{metrics['precision']:.3f}")
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                with col3:
                    st.metric("F1-Score", f"{metrics['f1-score']:.3f}")

        else:
            st.warning("No hay datos de prueba o predicciones disponibles")
            
    # Curva ROC y AUC solo para clasificación binaria
    if is_binary_classification and y_prob is not None:
        with tab3:
            with st.expander("Curva ROC (Receiver Operating Characteristic)"):
                st.markdown("""
            La **Curva ROC** es una representación gráfica que muestra la capacidad de un clasificador 
            para diferenciar entre clases. Se basa en dos métricas:
            
            - **Tasa de falsos positivos (False Positive Rate - FPR):** Proporción de negativos incorrectamente clasificados como positivos.
            - **Tasa de verdaderos positivos (True Positive Rate - TPR):** Proporción de positivos correctamente identificados.
            
            La curva muestra la relación entre TPR y FPR para diferentes umbrales de decisión.
            
            **Nota:** La curva ROC solo está disponible para problemas de clasificación binaria.
            """)
            
            try:
                # --- CORRECCIÓN PRINCIPAL: Convertir y_test a numérico ---
                # Crear mapeo de clases a números
                class_mapping = {class_name: i for i, class_name in enumerate(classes)}
                y_test_numeric = np.array([class_mapping[label] for label in y_test])
                
                # --- GRÁFICO: CURVA ROC ---
                st.subheader("Curva ROC")

                # Ajustar tamaño de figura
                fig_width = 10
                fig_height = 8
                fig_roc, ax_roc = plt.subplots(figsize=(fig_width, fig_height))

                # Clasificación binaria - CORREGIDO
                # Para clasificación binaria, usar la segunda clase como positiva
                pos_label = 1  # Segunda clase en el mapeo
                
                fpr, tpr, _ = roc_curve(y_test_numeric, y_prob[:, pos_label], pos_label=pos_label)
                roc_auc = auc(fpr, tpr)
                
                ax_roc.plot(fpr, tpr, color='darkorange', lw=3, label=f'Curva ROC (AUC = {roc_auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Línea base (AUC = 0.5)')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=11)
                ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=11)
                ax_roc.set_title('Curva ROC - Clasificación Binaria', fontsize=13, fontweight='bold')
                
                # Mostrar qué clase se considera positiva
                positive_class = classes[pos_label]
                negative_class = classes[0]
                ax_roc.text(0.02, 0.98, f'Clase positiva: {positive_class}\nClase negativa: {negative_class}', 
                           transform=ax_roc.transAxes, fontsize=10, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Leyenda fuera del gráfico
                ax_roc.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
                ax_roc.grid(True, alpha=0.3)
                
                # Ajustar tamaño de ticks
                ax_roc.tick_params(axis='both', which='major', labelsize=10)

                # Ajustar el layout para hacer espacio para la leyenda
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                st.pyplot(fig_roc)
                
                with st.expander("Área bajo la curva (AUC)"):
                    st.markdown("""
                El **AUC** cuantifica la calidad de la curva ROC en un solo valor:
                
                - **0.9 - 1.0:** Excelente poder discriminativo
                - **0.8 - 0.9:** Muy bueno
                - **0.7 - 0.8:** Aceptable
                - **0.6 - 0.7:** Pobre
                - **0.5 - 0.6:** No mejor que aleatorio
                - **< 0.5:** Peor que aleatorio
                """)
                        
                # --- MOSTRAR MÉTRICAS NUMÉRICAS ---
                st.subheader("Métricas de Evaluación AUC")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AUC Score", f"{roc_auc:.4f}")
                with col2:
                    st.metric("Interpretación", interpretar_auc(roc_auc))
                with col3:
                    quality = "✅ Excelente" if roc_auc >= 0.9 else "👍 Buena" if roc_auc >= 0.8 else "⚠️ Aceptable" if roc_auc >= 0.7 else "❌ Pobre"
                    st.metric("Calidad", quality)
                
                # Gráfico de métricas AUC
                fig_auc, ax_auc = plt.subplots(figsize=(8, 6))
                
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
                
                plt.tight_layout()
                st.pyplot(fig_auc)
                        
            except Exception as e:
                st.error(f"❌ Error al calcular las curvas ROC: {str(e)}")
                st.info("ℹ️ Esto puede ocurrir cuando hay problemas con las probabilidades predichas o las clases objetivo")
    
    # Mensaje informativo para problemas multiclase
    elif is_multiclass_classification:
        st.info("""
        **ℹ️ Información sobre Métricas para Clasificación Multiclase:**
        
        Para problemas de clasificación con más de 2 clases:
        - La **Curva ROC y AUC** no se muestran ya que son métricas diseñadas principalmente para clasificación binaria
        - En su lugar, se recomienda usar las métricas mostradas en el Reporte de Clasificación (Precisión, Recall, F1-Score)
        - La Matriz de Confusión proporciona una visión detallada del rendimiento por clase
        """)

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