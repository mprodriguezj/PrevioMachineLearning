import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

def ensemble_models_module(df):
    st.subheader("Selección de Modelo de Ensamble")
    
    # Explicación general sobre modelos de ensamble
    with st.expander("ℹ️ **Información sobre Modelos de Ensamble**"):
        st.markdown("""
        Los modelos de ensamble combinan múltiples algoritmos de aprendizaje para obtener 
        un mejor rendimiento predictivo que cualquiera de los algoritmos constituyentes por sí solo.
        
        **Ventajas principales:**
        - Reducen el sobreajuste (overfitting)
        - Mejoran la generalización
        - Son robustos frente a datos ruidosos
        """)
    
    # Selección de modelos para comparar
    st.write("**Selecciona los modelos a comparar:**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rf_selected = st.checkbox("Random Forest", value=True)
    with col2:
        ab_selected = st.checkbox("AdaBoost", value=True)
    with col3:
        gb_selected = st.checkbox("Gradient Boosting", value=True)
    with col4:
        bag_selected = st.checkbox("Bagging", value=True)
    
    selected_models = []
    if rf_selected:
        selected_models.append("Random Forest")
    if ab_selected:
        selected_models.append("AdaBoost")
    if gb_selected:
        selected_models.append("Gradient Boosting")
    if bag_selected:
        selected_models.append("Bagging")
    
    if not selected_models:
        st.warning("⚠️ Selecciona al menos un modelo para comparar")
        return
    
    # Data preparation
    st.write("**Preparación de datos:**")
    
    target_col = st.selectbox(
        "Selecciona la variable objetivo:",
        df.columns,
        key="target_select_ensemble"
    )
    
    available_features = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect(
        "Selecciona las variables predictoras:",
        available_features,
        default=available_features,
        key="features_select_ensemble"
    )
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    X = df[selected_features]
    y = df[target_col]
    X = pd.get_dummies(X)
    
    test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05, key="test_size_ensemble")
    random_state = st.number_input("Random state:", 0, 100, 42, key="random_state_ensemble")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Configuración de modelos seleccionados
    models_config = {}
    
    if "Random Forest" in selected_models:
        st.subheader("🌲 Random Forest")
        with st.expander("ℹ️ **Explicación de Random Forest**"):
            st.markdown("""
            **Random Forest** es un algoritmo de ensamble que combina múltiples árboles de decisión.
            
            **Cómo funciona:**
            1. Crea múltiples árboles de decisión con subconjuntos aleatorios de datos (bootstrapping)
            2. En cada división del árbol, considera solo un subconjunto aleatorio de características
            3. Combina las predicciones de todos los árboles (votación mayoritaria para clasificación)
            
            **Ventajas:**
            - Alta precisión
            - Resistente al sobreajuste
            - Maneja bien datos con muchas características
            """)
        models_config["Random Forest"] = configure_random_forest()
    
    if "AdaBoost" in selected_models:
        st.subheader("⚡ AdaBoost")
        with st.expander("ℹ️ **Explicación de AdaBoost**"):
            st.markdown("""
            **AdaBoost** (Adaptive Boosting) es un algoritmo de boosting que combina múltiples clasificadores débiles.
            
            **Cómo funciona:**
            1. Entrena secuencialmente múltiples modelos débiles (generalmente árboles poco profundos)
            2. Ajusta los pesos de las instancias, dando más peso a las mal clasificadas
            3. Combina todos los modelos débiles ponderando su contribución
            
            **Ventajas:**
            - Alta precisión
            - Menos propenso al sobreajuste que otros algoritmos
            - Automáticamente ajusta los pesos de las características
            """)
        models_config["AdaBoost"] = configure_adaboost()
    
    if "Gradient Boosting" in selected_models:
        st.subheader("📈 Gradient Boosting")
        with st.expander("ℹ️ **Explicación de Gradient Boosting**"):
            st.markdown("""
            **Gradient Boosting** es un algoritmo de boosting que optimiza una función de pérdida mediante descenso de gradiente.
            
            **Cómo funciona:**
            1. Construye modelos secuencialmente
            2. Cada nuevo modelo intenta corregir los errores del modelo anterior
            3. Utiliza el descenso de gradiente para minimizar una función de pérdida
            
            **Ventajas:**
            - Muy alta precisión
            - Flexible con diferentes funciones de pérdida
            - Maneja bien datos heterogéneos
            """)
        models_config["Gradient Boosting"] = configure_gradient_boosting()
    
    if "Bagging" in selected_models:
        st.subheader("👜 Bagging")
        with st.expander("ℹ️ **Explicación de Bagging**"):
            st.markdown("""
            **Bagging** (Bootstrap Aggregating) es una técnica que reduce la varianza de los algoritmos de aprendizaje.
            
            **Cómo funciona:**
            1. Crea múltiples subconjuntos de datos mediante muestreo con reemplazo (bootstrapping)
            2. Entrena un modelo en cada subconjunto
            3. Combina las predicciones de todos los modelos (promedio para regresión, votación para clasificación)
            
            **Ventajas:**
            - Reduce la varianza y ayuda a prevenir el sobreajuste
            - Funciona especialmente bien con algoritmos de alta varianza como árboles de decisión
            - Paralelizable (los modelos se entrenan independientemente)
            """)
        models_config["Bagging"] = configure_bagging()
    
    # Botón para entrenar todos los modelos seleccionados
    if st.button("🚀 Entrenar y Comparar Modelos", type="primary"):
        results = {}
        
        for model_name, config in models_config.items():
            with st.spinner(f"Entrenando {model_name}..."):
                try:
                    if model_name == "Random Forest":
                        model = RandomForestClassifier(**config, random_state=random_state)
                    elif model_name == "AdaBoost":
                        model = AdaBoostClassifier(**config, random_state=random_state)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingClassifier(**config, random_state=random_state)
                    elif model_name == "Bagging":
                        model = BaggingClassifier(**config, random_state=random_state)
                    
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    
                    results[model_name] = {
                        "model": model,
                        "y_pred": y_pred,
                        "y_prob": y_prob,
                        "classes": model.classes_
                    }
                    
                except Exception as e:
                    st.error(f"❌ Error al entrenar {model_name}: {str(e)}")
        
        if results:
            display_comparison_results(results, y_test)

def configure_random_forest():
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Número de árboles:", 10, 200, 100, key="rf_n_estimators")
        max_depth = st.slider("Profundidad máxima:", 1, 20, 5, key="rf_max_depth")
    
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2, key="rf_min_samples")
        criterion = st.selectbox("Criterio:", ["gini", "entropy"], key="rf_criterion")
    
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "criterion": criterion
    }

def configure_adaboost():
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 50, key="ab_n_estimators")
    
    with col2:
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01, key="ab_learning_rate")
    
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }

def configure_gradient_boosting():
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 100, key="gb_n_estimators")
        learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01, key="gb_learning_rate")
    
    with col2:
        max_depth = st.slider("Profundidad máxima:", 1, 10, 3, key="gb_max_depth")
        subsample = st.slider("Subsample:", 0.1, 1.0, 1.0, 0.1, key="gb_subsample")
    
    return {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample
    }

def configure_bagging():
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 100, 10, key="bag_n_estimators")
    
    with col2:
        max_samples = st.slider("Máximo samples:", 0.1, 1.0, 1.0, 0.1, key="bag_max_samples")
    
    return {
        "n_estimators": n_estimators,
        "max_samples": max_samples
    }

def display_comparison_results(results, y_test):
    st.subheader("📊 Comparación de Modelos")
    
    # Métricas de comparación
    comparison_data = []
    
    for model_name, result in results.items():
        report = classification_report(y_test, result["y_pred"], output_dict=True)
        accuracy = report['accuracy']
        weighted_avg = report['weighted avg']
        
        comparison_data.append({
            "Modelo": model_name,
            "Precisión": accuracy,
            "Recall": weighted_avg['recall'],
            "F1-Score": weighted_avg['f1-score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Mostrar tabla comparativa - FORMA CORRECTA
    # Aplicar formato solo a las columnas numéricas
    st.dataframe(
        comparison_df.style.format({
            'Precisión': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}'
        }).highlight_max(color='lightgreen').highlight_min(color='#ffcccb')
    )
    
    # Mostrar métricas individuales para cada modelo
    for model_name, result in results.items():
        with st.expander(f"📋 Métricas detalladas - {model_name}"):
            display_ensemble_results(
                y_test, 
                result["y_pred"], 
                result["y_prob"], 
                result["classes"], 
                model_name
            )

def display_ensemble_results(y_test, y_pred, y_prob, classes, model_name):
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curva ROC y AUC"])
    
    with tab1:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f'Matriz de Confusión - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
    
    with tab2:
        # Mejorar el reporte de clasificación como lo hicimos antes
        st.subheader("📋 Reporte de Clasificación")
        
        # Calcular el reporte
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Separar métricas por clase y generales - FORMA CORRECTA
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

    with tab3:
        if y_prob is not None:
            n_classes = len(classes)
            
            if n_classes == 2:
                # --- Caso binario ---
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {roc_auc:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Aleatorio (AUC = 0.5)')
                
                # Etiquetas y límites
                ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
                ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                
                ax.set_title(f'Curva ROC - {model_name}', fontsize=13, fontweight="bold")
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            else:
                # --- Caso multiclase ---
                y_test_bin = label_binarize(y_test, classes=classes)
                
                fpr, tpr, roc_auc = {}, {}, {}
                colors = sns.color_palette("husl", n_classes)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                            label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
                
                # Micro-promedio
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                ax.plot(fpr["micro"], tpr["micro"], color='black', linestyle=':', lw=3,
                        label=f'Micro-promedio (AUC = {roc_auc["micro"]:.3f})')
                
                # Línea base
                ax.plot([0, 1], [0, 1], 'k--', lw=2)
                
                # Etiquetas y límites
                ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
                ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                
                ax.set_title(f'Curva ROC Multiclase - {model_name}', fontsize=13, fontweight="bold")
                ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Mostrar métricas AUC agregadas
                macro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
                micro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='micro')
                st.write(f"**Macro AUC:** {macro_auc:.3f}")
                st.write(f"**Micro AUC:** {micro_auc:.3f}")
        else:
            st.info("Este modelo no soporta probabilidades de predicción")
