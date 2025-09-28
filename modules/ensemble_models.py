import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier,
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
)
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize

def ensemble_models_module(df):
    st.subheader("Selección de Modelos de Ensamble")
    
    # Información introductoria
    with st.expander("Acerca de los Modelos de Ensamble"):
        st.markdown("""
        Los modelos de ensamble combinan múltiples algoritmos de aprendizaje
        para lograr un mejor rendimiento predictivo que un único modelo.

        **Ventajas principales:**
        - Reducen el sobreajuste (overfitting)
        - Mejoran la generalización
        - Son más robustos frente a datos ruidosos
        """)
    
    # Selección de modelos
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
    
    # Preparación de datos
    st.write("**Preparación de datos:**")
    
    col1, col2 = st.columns(2)
    with col1:
        target_col = st.selectbox(
            "Selecciona la variable objetivo:",
            df.columns,
            key="target_select_ensemble"
        )
    
    with col2:
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Selecciona las variables predictoras:",
            available_features,
            default=available_features,
            key="features_select_ensemble"
        )
    
    # Resetear resultados si cambia la variable objetivo
    if 'previous_target' not in st.session_state:
        st.session_state.previous_target = target_col
    
    if st.session_state.previous_target != target_col:
        st.session_state.ensemble_results = None
        st.session_state.previous_target = target_col
        st.rerun()
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    X = df[selected_features]
    y = df[target_col]
    
    # Limpieza de datos
    missing_rows = X.isnull().any(axis=1) | y.isnull()
    if missing_rows.any():
        st.warning(f"⚠️ Se eliminaron {missing_rows.sum()} filas con valores nulos")
        X = X[~missing_rows]
        y = y[~missing_rows]
    
    # Variables categóricas
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

    # Variable objetivo - Detectar si es numérica o categórica
    is_numeric_target = False
    if y.dtype == 'object' or y.dtype.name == 'category':
        st.info("ℹ️ La variable objetivo es categórica - convirtiendo a numérico")
        try:
            y, uniques = pd.factorize(y)
            if len(uniques) < 2:
                st.error("❌ La variable objetivo debe tener al menos 2 categorías diferentes")
                return
        except Exception as e:
            st.error(f"❌ Error al procesar variable objetivo: {str(e)}")
            return
    else:
        # Es una variable numérica
        is_numeric_target = True
        st.info("ℹ️ La variable objetivo es numérica")
        # Verificar si tiene suficientes clases para clasificación
        unique_values = np.unique(y)
        if len(unique_values) < 2:
            st.error("❌ La variable objetivo debe tener al menos 2 valores diferentes")
            return
        elif len(unique_values) > 10:
            st.warning("⚠️ La variable objetivo tiene muchos valores únicos. Considera si es apropiado para clasificación.")
            
        # Opciones para manejar variables numéricas
        st.subheader("Opciones para variable objetivo numérica")
        numeric_option = st.radio(
            "¿Cómo deseas manejar la variable objetivo numérica?",
            ["Discretizar en intervalos para clasificación", "Usar modelo de regresión"],
            key="numeric_option_ensemble"
        )
        
        if numeric_option == "Discretizar en intervalos para clasificación":
            # Opciones de discretización
            discretize_method = st.selectbox(
                "Método de discretización:",
                ["Intervalos iguales", "Cuantiles", "Personalizado"],
                key="discretize_method_ensemble"
            )
            
            if discretize_method == "Intervalos iguales":
                n_bins = st.slider("Número de intervalos:", 2, 10, 4, key="n_bins_equal_ensemble")
                # Discretizar en intervalos iguales
                bins = np.linspace(y.min(), y.max(), n_bins + 1)
                labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
                y_discretized = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
                
                # Mostrar distribución de clases
                class_counts = pd.Series(y_discretized).value_counts().sort_index()
                st.write("Distribución de clases después de discretización:")
                st.bar_chart(class_counts)
                
                # Reemplazar la variable objetivo con la versión discretizada
                y, uniques = pd.factorize(y_discretized)
                st.success(f"✅ Variable objetivo discretizada en {n_bins} intervalos iguales")
                is_numeric_target = False
                
            elif discretize_method == "Cuantiles":
                n_bins = st.slider("Número de cuantiles:", 2, 10, 4, key="n_bins_quantile_ensemble")
                # Discretizar por cuantiles
                y_discretized = pd.qcut(y, q=n_bins, labels=False)
                
                # Mostrar distribución de clases
                class_counts = pd.Series(y_discretized).value_counts().sort_index()
                st.write("Distribución de clases después de discretización:")
                st.bar_chart(class_counts)
                
                # Reemplazar la variable objetivo con la versión discretizada
                y = y_discretized
                st.success(f"✅ Variable objetivo discretizada en {n_bins} cuantiles")
                is_numeric_target = False
                
            elif discretize_method == "Personalizado":
                # Permitir al usuario definir puntos de corte personalizados
                min_val = float(y.min())
                max_val = float(y.max())
                
                st.write(f"Rango de valores: {min_val:.2f} a {max_val:.2f}")
                
                # Entrada de texto para puntos de corte
                cutpoints_text = st.text_input(
                    "Puntos de corte (separados por coma):",
                    value=f"{min_val},{(min_val+max_val)/2:.2f},{max_val}",
                    key="custom_cutpoints_ensemble"
                )
                
                try:
                    # Convertir texto a lista de puntos de corte
                    cutpoints = [float(x.strip()) for x in cutpoints_text.split(",")]
                    cutpoints = sorted(list(set(cutpoints)))  # Eliminar duplicados y ordenar
                    
                    if len(cutpoints) < 2:
                        st.error("❌ Se necesitan al menos 2 puntos de corte")
                    else:
                        # Asegurarse de que los puntos de corte cubran todo el rango
                        if cutpoints[0] > min_val:
                            cutpoints.insert(0, min_val)
                        if cutpoints[-1] < max_val:
                            cutpoints.append(max_val)
                        
                        # Crear etiquetas y discretizar
                        labels = [f'{cutpoints[i]:.2f}-{cutpoints[i+1]:.2f}' for i in range(len(cutpoints)-1)]
                        y_discretized = pd.cut(y, bins=cutpoints, labels=labels, include_lowest=True)
                        
                        # Mostrar distribución de clases
                        class_counts = pd.Series(y_discretized).value_counts().sort_index()
                        st.write("Distribución de clases después de discretización:")
                        st.bar_chart(class_counts)
                        
                        # Reemplazar la variable objetivo con la versión discretizada
                        y, uniques = pd.factorize(y_discretized)
                        st.success(f"✅ Variable objetivo discretizada con puntos de corte personalizados")
                        is_numeric_target = False
                        
                except Exception as e:
                    st.error(f"❌ Error al procesar puntos de corte: {str(e)}")

    # Distribución de clases después de limpieza
    class_counts = pd.Series(y).value_counts()
    
    # Validación final
    if len(np.unique(y)) < 2:
        st.error("❌ No hay suficientes clases después de la limpieza. Se necesitan al menos 2 clases diferentes.")
        return
    
    # División del dataset
    st.write("**Configuración de división de datos:**")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05, key="test_size_ensemble")
    with col2:
        random_state = st.number_input("Random state:", 0, 100, 42, key="random_state_ensemble")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        st.success("✅ División de datos realizada exitosamente")
    except Exception as e:
        st.error(f"❌ Error en la división de datos: {str(e)}")
        return
    
    # Configuración de modelos con explicaciones
    models_config = {}
    
    if "Random Forest" in selected_models:
        st.subheader("Random Forest")
        with st.expander("Explicación de Random Forest"):
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
        st.subheader("AdaBoost")
        with st.expander("Explicación de AdaBoost"):
            st.markdown("""
            **AdaBoost** (Adaptive Boosting) es un algoritmo de boosting que combina múltiples clasificadores débiles.
            
            **Cómo funciona:**
            1. Entrena secuencialmente múltiples modelos débils (generalmente árboles poco profundos)
            2. Ajusta los pesos de las instancias, dando más peso a las mal clasificadas
            3. Combina todos los modelos débiles ponderando su contribución
            
            **Ventajas:**
            - Alta precisión
            - Menos propenso al sobreajuste que otros algoritmos
            - Automáticamente ajusta los pesos de las características
            """)
        models_config["AdaBoost"] = configure_adaboost()
    
    if "Gradient Boosting" in selected_models:
        st.subheader("Gradient Boosting")
        with st.expander("Explicación de Gradient Boosting"):
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
        st.subheader("Bagging")
        with st.expander("Explicación de Bagging"):
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
    
    # Entrenamiento
    if st.button("Entrenar y Comparar Modelos", type="primary"):
        results = train_models(models_config, X_train, y_train, X_test, y_test, random_state, is_numeric_target)
        if results:
            st.session_state.ensemble_results = results
            st.session_state.ensemble_y_test = y_test
            st.session_state.is_numeric_target = is_numeric_target  # Guardar si es numérica
            st.success("✅ Modelos entrenados exitosamente")
        else:
            st.error("❌ No se pudo entrenar ningún modelo. Revisa parámetros y datos.")
    
    # Mostrar resultados si existen
    if 'ensemble_results' in st.session_state and st.session_state.ensemble_results:
        display_comparison_results(
            st.session_state.ensemble_results, 
            st.session_state.ensemble_y_test,
            st.session_state.is_numeric_target
        )

def configure_random_forest():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de árboles:", 10, 200, 100, key="rf_n_estimators")
        max_depth = st.slider("Profundidad máxima:", 1, 20, None, key="rf_max_depth")
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2, key="rf_min_samples")
        criterion = st.selectbox("Criterio:", ["gini", "entropy"], key="rf_criterion")
    return {"n_estimators": n_estimators, "max_depth": max_depth,
            "min_samples_split": min_samples_split, "criterion": criterion}

def configure_adaboost():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 50, key="ab_n_estimators")
    with col2:
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01, key="ab_learning_rate")
    return {"n_estimators": n_estimators, "learning_rate": learning_rate}

def configure_gradient_boosting():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 100, key="gb_n_estimators")
        learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01, key="gb_learning_rate")
    with col2:
        max_depth = st.slider("Profundidad máxima:", 1, 10, 3, key="gb_max_depth")
        min_samples_split = st.slider("Mínimo samples split:", 2, 20, 2, key="gb_min_samples")
    return {"n_estimators": n_estimators, "learning_rate": learning_rate,
            "max_depth": max_depth, "min_samples_split": min_samples_split}

def configure_bagging():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 100, 10, key="bag_n_estimators")
    with col2:
        max_samples = st.slider("Máximo samples:", 0.1, 1.0, 1.0, 0.1, key="bag_max_samples")
    return {"n_estimators": n_estimators, "max_samples": max_samples}

def train_models(models_config, X_train, y_train, X_test, y_test, random_state, is_numeric_target=False):
    results = {}
    for model_name, config in models_config.items():
        with st.spinner(f"Entrenando {model_name}..."):
            try:
                # Seleccionar el modelo adecuado según el tipo de variable objetivo
                if is_numeric_target:
                    # Modelos de regresión para variables numéricas
                    if model_name == "Random Forest":
                        model = RandomForestRegressor(**config, random_state=random_state)
                    elif model_name == "AdaBoost":
                        model = AdaBoostRegressor(**config, random_state=random_state)
                    elif model_name == "Gradient Boosting":
                        model = GradientBoostingRegressor(**config, random_state=random_state)
                    elif model_name == "Bagging":
                        model = BaggingRegressor(**config, random_state=random_state)
                else:
                    # Modelos de clasificación para variables categóricas
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
                
                # Solo los modelos de clasificación tienen predict_proba y classes_
                if is_numeric_target:
                    results[model_name] = {
                        "y_pred": y_pred,
                        "model": model
                    }
                else:
                    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    results[model_name] = {
                        "y_pred": y_pred,
                        "y_prob": y_prob,
                        "classes": model.classes_
                    }
                    
                st.success(f"✅ {model_name} entrenado exitosamente")

            except Exception as e:
                st.error(f"❌ Error al entrenar {model_name}: {str(e)}")
    
    return results

def display_comparison_results(results, y_test, is_numeric_target):
    st.subheader("Comparación de Modelos")
    
    comparison_data = []
    
    if is_numeric_target:
        # Métricas para modelos de regresión
        for model_name, result in results.items():
            y_pred = result["y_pred"]
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            comparison_data.append({
                "Modelo": model_name,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Definir explícitamente las columnas numéricas para el resaltado
        numeric_columns = ['MSE', 'RMSE', 'MAE', 'R²']
        
        # Para MSE, RMSE y MAE, los valores más bajos son mejores
        min_better = ['MSE', 'RMSE', 'MAE']
        # Para R², los valores más altos son mejores
        max_better = ['R²']
        
        # Aplicar formato y resaltado
        styled_df = comparison_df.style.format({
            'MSE': '{:.3f}', 
            'RMSE': '{:.3f}', 
            'MAE': '{:.3f}',
            'R²': '{:.3f}'
        })
        
        # Resaltar los mejores valores (mínimos para errores, máximos para R²)
        for col in min_better:
            styled_df = styled_df.highlight_min(
                subset=[col], 
                color='lightgreen',
                axis=0
            )
        
        for col in max_better:
            styled_df = styled_df.highlight_max(
                subset=[col], 
                color='lightgreen',
                axis=0
            )
    else:
        # Métricas para modelos de clasificación
        for model_name, result in results.items():
            report = classification_report(y_test, result["y_pred"], output_dict=True)
            accuracy = report['accuracy']
            weighted_avg = report['weighted avg']
            comparison_data.append({
                "Modelo": model_name,
                "Accuracy": accuracy,
                "Recall": weighted_avg['recall'],
                "F1-Score": weighted_avg['f1-score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Definir explícitamente las columnas numéricas para el resaltado
        numeric_columns = ['Accuracy', 'Recall', 'F1-Score']
        
        # Aplicar formato y resaltado SOLO a las columnas numéricas
        styled_df = comparison_df.style.format({
            'Accuracy': '{:.3f}', 
            'Recall': '{:.3f}', 
            'F1-Score': '{:.3f}'
        }).highlight_max(
            subset=numeric_columns, 
            color='lightgreen',
            axis=0  # Buscar máximo por columna
        ).highlight_min(
            subset=numeric_columns, 
            color='#ffcccb',
            axis=0  # Buscar mínimo por columna
        )
    
    st.dataframe(styled_df, use_container_width=True)

    # Resultados individuales
    for model_name, result in results.items():
        with st.expander(f"Resultados detallados - {model_name}"):
            if is_numeric_target:
                show_regression_results(
                    y_test, 
                    result["y_pred"], 
                    model_name
                )
            else:
                show_model_results(
                    y_test, 
                    result["y_pred"], 
                    result["y_prob"], 
                    result["classes"], 
                    model_name, 
                    is_numeric_target
                )

def show_regression_results(y_test, y_pred, model_name):
    """Muestra resultados para modelos de regresión"""
    tab1, tab2 = st.tabs(["Métricas de Error", "Visualizaciones"])
    
    with tab1:
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mostrar métricas en formato de tabla
        metrics_df = pd.DataFrame({
            'Métrica': ['MSE', 'RMSE', 'MAE', 'R²'],
            'Valor': [mse, rmse, mae, r2]
        })
        
        st.table(metrics_df.style.format({'Valor': '{:.4f}'}))
        
        # Explicaciones de las métricas
        with st.expander("Explicación de las métricas"):
            st.markdown("""
            - **MSE (Error Cuadrático Medio)**: Promedio de los errores al cuadrado. Penaliza errores grandes.
            - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del MSE. Está en las mismas unidades que la variable objetivo.
            - **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos. Menos sensible a valores atípicos que MSE.
            - **R² (Coeficiente de Determinación)**: Proporción de la varianza explicada por el modelo. 1 es perfecto, 0 es igual al modelo base, negativo es peor que el modelo base.
            """)
    
    with tab2:
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gráfico de dispersión: Valores reales vs predicciones
        ax1.scatter(y_test, y_pred, alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax1.set_xlabel('Valores Reales')
        ax1.set_ylabel('Predicciones')
        ax1.set_title(f'{model_name}: Predicciones vs Valores Reales')
        
        # Histograma de residuos
        residuos = y_test - y_pred
        ax2.hist(residuos, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Residuos')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Residuos')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Explicación de los gráficos
        with st.expander("Interpretación de los gráficos"):
            st.markdown("""
            **Predicciones vs Valores Reales**:
            - Los puntos deben estar cerca de la línea roja diagonal para un buen modelo.
            - Puntos dispersos indican mayor error en las predicciones.
            
            **Distribución de Residuos**:
            - Idealmente debe ser simétrica alrededor de cero (línea roja).
            - Una distribución sesgada puede indicar que el modelo tiene un sesgo sistemático.
            - Valores extremos pueden indicar outliers o casos donde el modelo tiene dificultades.
            """)

def show_model_results(y_test, y_pred, y_prob, classes, model_name, is_numeric_target):
    # Para variables categóricas, mostrar todas las pestañas
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curva ROC y AUC"])

    with tab1:
        show_confusion_matrix(y_test, y_pred, classes, model_name)
    
    with tab2:
        show_classification_report(y_test, y_pred, model_name)
    
    with tab3:
        show_roc_curve(y_test, y_prob, classes, model_name)

def show_confusion_matrix(y_test, y_pred, classes, model_name):
    """Muestra matriz de confusión con tamaño de fuente ajustado"""
    cm = confusion_matrix(y_test, y_pred)
    
    # Ajustar tamaño de figura según número de clases
    n_classes = len(classes)
    fig_size = max(8, n_classes * 0.8)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Tamaño de fuente ajustable
    font_size = max(8, 12 - n_classes//2)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={'size': font_size, 'weight': 'bold'},
                cbar_kws={'shrink': 0.8})
    
    ax.set_title(f"Matriz de Confusión - {model_name}", fontsize=12)
    ax.set_xlabel('Predicciones', fontsize=10)
    ax.set_ylabel('Valores Reales', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Rotar etiquetas si hay muchas clases
    if n_classes > 5:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    st.pyplot(fig)

def show_classification_report(y_test, y_pred, model_name):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    accuracy = report['accuracy']
    weighted_avg = report.get('weighted avg', {})

    st.write("**Métricas Principales:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Exactitud", f"{accuracy:.3f}")
    col2.metric("Precisión", f"{weighted_avg.get('precision', 0):.3f}")
    col3.metric("Recall", f"{weighted_avg.get('recall', 0):.3f}")
    col4.metric("F1-Score", f"{weighted_avg.get('f1-score', 0):.3f}")

    st.write("**Métricas por Clase:**")
    st.dataframe(report_df.style.format({
        'precision': '{:.3f}', 'recall': '{:.3f}',
        'f1-score': '{:.3f}', 'support': '{:.0f}'
    }), use_container_width=True)

def show_roc_curve(y_test, y_prob, classes, model_name):
    if y_prob is not None and len(classes) > 1:
        try:
            n_classes = len(classes)
            fig, ax = plt.subplots(figsize=(10, 6))

            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
            else:
                y_test_bin = label_binarize(y_test, classes=classes)
                colors = sns.color_palette("husl", n_classes)
                for i, color in zip(range(n_classes), colors):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=color, lw=2,
                            label=f'Clase {classes[i]} (AUC = {roc_auc:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', label='Línea base')
            ax.set_xlabel("Tasa de Falsos Positivos")
            ax.set_ylabel("Tasa de Verdaderos Positivos")
            ax.set_title(f"Curva ROC - {model_name}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar la curva ROC: {str(e)}")
    else:
        st.info("ℹ️ La curva ROC no está disponible para este modelo")