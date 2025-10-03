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
    
    # --- INICIALIZACIÓN DE ESTADO ---
    if 'ensemble_current_state' not in st.session_state:
        st.session_state.ensemble_current_state = {}
    
    # --- DETECCIÓN DE CAMBIOS ---
    current_config = {
        'target_col': None,
        'selected_features': None,
        'cleaning_strategy': None,
        'numeric_option': None,
        'discretize_method': None,
        'n_bins': None,
        'test_size': None,
        'random_state': None,
        'selected_models': None,
        'rf_params': None,
        'ab_params': None,
        'gb_params': None,
        'bag_params': None
    }
    
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
        rf_selected = st.checkbox("Random Forest", value=True, key="rf_checkbox")
    with col2:
        ab_selected = st.checkbox("AdaBoost", value=True, key="ab_checkbox")
    with col3:
        gb_selected = st.checkbox("Gradient Boosting", value=True, key="gb_checkbox")
    with col4:
        bag_selected = st.checkbox("Bagging", value=True, key="bag_checkbox")
    
    selected_models = []
    if rf_selected:
        selected_models.append("Random Forest")
    if ab_selected:
        selected_models.append("AdaBoost")
    if gb_selected:
        selected_models.append("Gradient Boosting")
    if bag_selected:
        selected_models.append("Bagging")
    
    current_config['selected_models'] = tuple(selected_models)
    
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
        current_config['target_col'] = target_col
    
    with col2:
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Selecciona las variables predictoras:",
            available_features,
            default=available_features,
            key="features_select_ensemble"
        )
        current_config['selected_features'] = tuple(selected_features)
    
    if not selected_features:
        st.warning("⚠️ Selecciona al menos una variable predictora")
        return
    
    # Construir X, y
    X = df[selected_features]
    y = df[target_col]
    
    # --- Limpieza y procesamiento de datos MEJORADO ---
    st.write("**Limpieza y procesamiento de datos:**")

    rows_before = len(X)

    # Verificar si hay valores infinitos o nulos
    has_inf = np.any(np.isinf(X.select_dtypes(include=['float64', 'int64']).values), axis=1).any()
    has_nulls_X = X.isnull().any().any()
    has_nulls_y = y.isnull().any()

    # Solo mostrar opciones de limpieza si hay valores problemáticos
    if has_inf or has_nulls_X or has_nulls_y:
        cleaning_strategy = st.selectbox(
            "Estrategia para manejar valores faltantes/infinitos:",
            ["Eliminar filas afectadas", "Rellenar con valores apropiados"],
            help="Eliminar: Más conservador | Rellenar: Preserva más datos",
            key="cleaning_strategy_ensemble"
        )
        current_config['cleaning_strategy'] = cleaning_strategy
    else:
        cleaning_strategy = "Eliminar filas afectadas"
        st.info("ℹ️ No se detectaron valores infinitos ni nulos en el dataset")

    # 1. Manejar valores infinitos - solo si existen
    inf_mask = np.any(np.isinf(X.select_dtypes(include=['float64', 'int64']).values), axis=1)
    if inf_mask.any():
        st.warning(f"⚠️ Se detectaron {inf_mask.sum()} filas con valores infinitos.")
        
        if cleaning_strategy == "Eliminar filas afectadas":
            X = X[~inf_mask]
            y = y[~inf_mask]
            st.success("✅ Filas con infinitos ELIMINADAS")
        else:
            # Reemplazar infinitos por NaN para luego imputar
            X = X.replace([np.inf, -np.inf], np.nan)
            st.info("ℹ️ Infinitos convertidos a NaN para imputación")

    # 2. Manejar valores nulos - solo si existen
    total_missing_X = X.isnull().any(axis=1).sum()
    total_missing_y = y.isnull().sum()

    # Solo mostrar resumen si hay nulos
    if total_missing_X > 0 or total_missing_y > 0:
        st.write(f"**Resumen de valores nulos:**")
        if total_missing_X > 0:
            st.write(f"- Filas con nulos en predictores (X): {total_missing_X}")
        if total_missing_y > 0:
            st.write(f"- Filas con nulos en objetivo (y): {total_missing_y}")

    # Estrategia diferente para X vs y - solo aplicar si hay nulos
    if has_nulls_X or has_nulls_y:
        if cleaning_strategy == "Eliminar filas afectadas":
            # ELIMINAR: Más seguro para el modelo
            missing_mask = X.isnull().any(axis=1) | y.isnull()
            if missing_mask.any():
                rows_removed = missing_mask.sum()
                X = X[~missing_mask]
                y = y[~missing_mask]
                st.success(f"✅ Se eliminaron {rows_removed} filas con valores nulos")
                
        else:
            # RELLENAR: Preserva datos
            # Para X: Imputar por tipo de variable
            numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            # Rellenar numéricos con mediana (más robusta que media)
            for col in numeric_cols:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
                    st.info(f"ℹ️ Numérico '{col}': nulos rellenados con mediana ({X[col].median():.2f})")
            
            # Rellenar categóricos con moda
            for col in categorical_cols:
                if X[col].isna().any():
                    mode_val = X[col].mode()[0] if not X[col].mode().empty else "DESCONOCIDO"
                    X[col] = X[col].fillna(mode_val)
                    st.info(f"ℹ️ Categórico '{col}': nulos rellenados con moda ('{mode_val}')")
            
            # Para y: Solo eliminar (crítico para el objetivo)
            if y.isnull().any():
                y_null_count = y.isnull().sum()
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]
                st.success(f"✅ Se eliminaron {y_null_count} filas con nulos en variable objetivo (CRÍTICO)")

    # 3. Verificación final - solo mostrar si hubo cambios
    rows_after = len(X)
    rows_removed_total = rows_before - rows_after

    if rows_removed_total > 0:
        removal_percentage = (rows_removed_total / rows_before) * 100
        st.warning(f"⚠️ Resumen final: {rows_removed_total} filas removidas ({removal_percentage:.1f}%)")
        
        if removal_percentage > 30:
            st.error("❌ ¡Alto porcentaje de datos perdidos! Considera revisar tu dataset")
        elif removal_percentage > 10:
            st.warning("⚠️ Porcentaje moderado de datos perdidos")
    else:
        st.info("ℹ️ No se removieron filas durante la limpieza")

    st.info(f"ℹ️ Filas restantes para modelo: {rows_after}")

    if len(X) < 10:
        st.error("❌ Dataset demasiado pequeño después de limpieza. No se puede continuar.")
        return
    elif len(X) < 50:
        st.warning("⚠️ Dataset muy pequeño. Los resultados pueden no ser confiables.")
        
    # Detectar variables categóricas (solo explícitas - objetos y categorías)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        st.info(f"ℹ️ Variables categóricas detectadas: {', '.join(categorical_cols)}")
        
        # Convertir a dummies con manejo de errores más robusto
        try:
            # Crear una copia de seguridad de X
            X_backup = X.copy()
            
            # Convertir solo las columnas categóricas
            X_cat = pd.get_dummies(X[categorical_cols], drop_first=True, dummy_na=False)
            
            # Mantener las columnas numéricas que no son categóricas
            non_cat_cols = [col for col in X.columns if col not in categorical_cols]
            X_non_cat = X[non_cat_cols]
            
            # Combinar ambos conjuntos de datos
            X = pd.concat([X_non_cat, X_cat], axis=1)
            
            st.success(f"✅ Variables categóricas convertidas a one-hot encoding. Nuevas dimensiones: {X.shape}")
        except Exception as e:
            st.error(f"❌ Error al convertir variables categóricas: {str(e)}")
            st.warning("Intentando método alternativo de conversión...")
            
            try:
                # Restaurar X desde la copia de seguridad
                X = X_backup.copy()
                
                # Método alternativo: convertir una por una
                for col in categorical_cols:
                    try:
                        # Convertir a string primero para manejar diferentes tipos de datos
                        X[col] = X[col].astype(str)
                    except Exception as e_col:
                        st.warning(f"No se pudo convertir la columna '{col}': {str(e_col)}")
                
                # Intentar la conversión con manejo explícito de columnas
                dummies_list = []
                remaining_cols = []
                
                for col in X.columns:
                    if col in categorical_cols:
                        try:
                            # Crear dummies para esta columna
                            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                            dummies_list.append(dummies)
                        except Exception as e_dummy:
                            st.warning(f"Error al crear dummies para '{col}': {str(e_dummy)}")
                            # Mantener la columna original si falla
                            remaining_cols.append(col)
                    else:
                        remaining_cols.append(col)
                
                # Reconstruir el DataFrame
                if dummies_list:
                    all_dummies = pd.concat(dummies_list, axis=1)
                    X = pd.concat([X[remaining_cols], all_dummies], axis=1)
                    st.success(f"✅ Variables convertidas con método alternativo. Nuevas dimensiones: {X.shape}")
                else:
                    st.warning("⚠️ No se pudieron crear variables dummy. Manteniendo variables originales.")
            except Exception as e2:
                st.error(f"❌ Error en método alternativo: {str(e2)}")
                st.warning("⚠️ Continuando con las variables originales sin convertir a dummies.")
    else:
        st.info("ℹ️ No se detectaron variables categóricas en los predictores")
        st.info("✅ **One-hot encoding:** No requerido - no hay variables categóricas")
    
    # Procesamiento de variable objetivo
    # Detectar si la variable objetivo es numérica o categórica
    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    
    # Variable para controlar si usamos regresión o clasificación
    use_regression_model = True  # Valor por defecto
    
    if is_numeric_target:
        st.info(f"ℹ️ Variable objetivo '{target_col}' detectada como numérica/continua")
        
        # Para variables numéricas, verificamos valores nulos y outliers
        if y.isna().any():
            st.warning(f"⚠️ Se eliminaron {y.isna().sum()} filas con valores nulos en la variable objetivo")
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
        
        # Detectar y manejar outliers (opcional)
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (y < lower_bound) | (y > upper_bound)
        
        if outliers.sum() > 0:
            st.info(f"ℹ️ Se detectaron {outliers.sum()} valores atípicos en la variable objetivo")
        
        # Opciones para manejar variables numéricas
        st.subheader("Opciones para variable objetivo numérica")
        numeric_option = st.radio(
            "¿Cómo deseas manejar la variable objetivo numérica?",
            ["Discretizar en intervalos para clasificación", "Usar modelo de regresión"],
            key="numeric_option_ensemble"
        )
        current_config['numeric_option'] = numeric_option
        
        if numeric_option == "Discretizar en intervalos para clasificación":
            use_regression_model = False
            # Opciones de discretización
            discretize_method = st.selectbox(
                "Método de discretización:",
                ["Intervalos iguales", "Cuantiles", "Personalizado"],
                key="discretize_method_ensemble"
            )
            current_config['discretize_method'] = discretize_method
            
            if discretize_method == "Intervalos iguales":
                n_bins = st.slider("Número de intervalos:", 2, 10, 4, key="n_bins_equal_ensemble")
                current_config['n_bins'] = n_bins
                # Discretizar en intervalos iguales
                bins = np.linspace(y.min(), y.max(), n_bins + 1)
                labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
                y_discretized = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
                
                # Mostrar distribución de clases
                class_counts = y_discretized.value_counts().sort_index()
                st.write("Distribución de clases después de discretización:")
                st.bar_chart(class_counts)
                
                # Reemplazar la variable objetivo con la versión discretizada
                y = y_discretized
                st.success(f"✅ Variable objetivo discretizada en {n_bins} intervalos iguales")
                is_numeric_target = False
                
            elif discretize_method == "Cuantiles":
                n_bins = st.slider("Número de cuantiles:", 2, 10, 4, key="n_bins_quantile_ensemble")
                current_config['n_bins'] = n_bins
                # Discretizar por cuantiles
                bins = pd.qcut(y, q=n_bins, retbins=True)[1]
                labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
                y_discretized = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
                
                # Mostrar distribución de clases
                class_counts = y_discretized.value_counts().sort_index()
                st.write("Distribución de clases después de discretización:")
                st.bar_chart(class_counts)
                
                # Reemplazar la variable objetivo con la versión discretizada
                y = y_discretized
                st.success(f"✅ Variable objetivo discretizada en {n_bins} cuantiles")
                is_numeric_target = False
                
            elif discretize_method == "Personalizado":
                use_regression_model = False
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
                        class_counts = y_discretized.value_counts().sort_index()
                        st.write("Distribución de clases después de discretización:")
                        st.bar_chart(class_counts)
                        
                        # Reemplazar la variable objetivo con la versión discretizada
                        y = y_discretized
                        st.success(f"✅ Variable objetivo discretizada con puntos de corte personalizados")
                        is_numeric_target = False
                        
                except Exception as e:
                    st.error(f"❌ Error al procesar puntos de corte: {str(e)}")
        else:
            # Para árboles de decisión con variable objetivo numérica, usamos DecisionTreeRegressor
            use_regression_model = True
            st.info("ℹ️ Se utilizará un modelo de regresión para la variable objetivo numérica")
        
    else:
        st.info(f"ℹ️ Variable objetivo '{target_col}' detectada como categórica")
        use_regression_model = False
        
        # Para variables categóricas, limpiamos y procesamos
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
        
        # Validación final para clasificación
        if len(y.unique()) < 2:
            st.error("❌No hay suficientes clases para entrenar el modelo. Se necesitan al menos 2 clases diferentes.")
            return
    
    # --- Configuración de división de datos ---
    st.write("**Configuración de división de datos:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Tamaño del conjunto de prueba:", 0.1, 0.4, 0.2, 0.05,
                            key="test_size_ensemble")
        current_config['test_size'] = test_size
    
    with col2:
        random_state = st.number_input("Random state:", 0, 100, 42, key="random_state_ensemble")
        current_config['random_state'] = random_state
    
    # Validar estratificación
    can_stratify = len(y.unique()) >= 2 and y.value_counts().min() >= 2
    
    # División del dataset
    try:
        if can_stratify and not use_regression_model:
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
    
    # --- Configuración de modelos con explicaciones ---
    models_config = {}
    
    if "Random Forest" in selected_models:
        st.subheader("Random Forest")
        with st.expander("Explicación de Random Forest"):
            if use_regression_model:
                st.markdown("""
                **Random Forest para Regresión** combina múltiples árboles de regresión.
                
                **Cómo funciona:**
                1. Crea múltiples árboles de regresión con subconjuntos aleatorios de datos
                2. En cada división del árbol, considera solo un subconjunto aleatorio de características
                3. Combina las predicciones de todos los árboles (promedio para regresión)
                
                **Ventajas:**
                - Alta precisión en problemas de regresión
                - Resistente al sobreajuste
                - Maneja bien datos con muchas características
                """)
            else:
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
        rf_config = configure_random_forest(use_regression_model)
        models_config["Random Forest"] = rf_config
        current_config['rf_params'] = rf_config
    
    if "AdaBoost" in selected_models:
        st.subheader("AdaBoost")
        with st.expander("Explicación de AdaBoost"):
            if use_regression_model:
                st.markdown("""
                **AdaBoost para Regresión** adapta el algoritmo boosting a problemas de regresión.
                
                **Cómo funciona:**
                1. Entrena secuencialmente múltiples regresores débiles
                2. Ajusta los pesos de las instancias, dando más peso a las predicciones con mayor error
                3. Combina todos los regresores ponderando su contribución
                
                **Ventajas:**
                - Buena precisión en regresión
                - Menos propenso al sobreajuste
                - Automáticamente ajusta los pesos de las características
                """)
            else:
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
        ab_config = configure_adaboost(use_regression_model)
        models_config["AdaBoost"] = ab_config
        current_config['ab_params'] = ab_config
    
    if "Gradient Boosting" in selected_models:
        st.subheader("Gradient Boosting")
        with st.expander("Explicación de Gradient Boosting"):
            if use_regression_model:
                st.markdown("""
                **Gradient Boosting para Regresión** optimiza funciones de pérdida para problemas de regresión.
                
                **Cómo funciona:**
                1. Construye modelos secuencialmente
                2. Cada nuevo modelo intenta corregir los errores residuales del modelo anterior
                3. Utiliza el descenso de gradiente para minimizar una función de pérdida (MSE, MAE, etc.)
                
                **Ventajas:**
                - Muy alta precisión en regresión
                - Flexible con diferentes funciones de pérdida
                - Maneja bien datos heterogéneos
                """)
            else:
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
        gb_config = configure_gradient_boosting(use_regression_model)
        models_config["Gradient Boosting"] = gb_config
        current_config['gb_params'] = gb_config
    
    if "Bagging" in selected_models:
        st.subheader("Bagging")
        with st.expander("Explicación de Bagging"):
            if use_regression_model:
                st.markdown("""
                **Bagging para Regresión** reduce la varianza en algoritmos de regresión.
                
                **Cómo funciona:**
                1. Crea múltiples subconjuntos de datos mediante muestreo con reemplazo (bootstrapping)
                2. Entrena un modelo de regresión en cada subconjunto
                3. Combina las predicciones de todos los modelos (promedio para regresión)
                
                **Ventajas:**
                - Reduce la varianza y ayuda a prevenir el sobreajuste
                - Funciona especialmente bien con algoritmos de alta varianza
                - Paralelizable (los modelos se entrenan independientemente)
                """)
            else:
                st.markdown("""
                **Bagging** (Bootstrap Aggregating) es una técnica que reduce la varianza de los algoritmos de aprendizaje.
                
                **Cómo funciona:**
                1. Crea múltiples subconjuntos de datos mediante muestreo con reemplazo (bootstrapping)
                2. Entrena un modelo en cada subconjunto
                3. Combina las predicciones de todos los modelos (votación para clasificación)
                
                **Ventajas:**
                - Reduce la varianza y ayuda a prevenir el sobreajuste
                - Funciona especialmente bien con algoritmos de alta varianza como árboles de decisión
                - Paralelizable (los modelos se entrenan independientemente)
                """)
        bag_config = configure_bagging()
        models_config["Bagging"] = bag_config
        current_config['bag_params'] = bag_config
    
    # --- DETECCIÓN DE CAMBIOS Y LIMPIEZA DE RESULTADOS ---
    config_changed = False
    if hasattr(st.session_state, 'ensemble_previous_state'):
        # Comparar configuración actual con anterior
        for key in current_config:
            if current_config[key] != st.session_state.ensemble_previous_state.get(key):
                config_changed = True
                break
    
    # Si la configuración cambió, limpiar resultados anteriores
    if config_changed:
        st.session_state.ensemble_results = None
        st.session_state.ensemble_trained_models = None
        st.info("🔄 Configuración modificada. Entrena los modelos nuevamente.")
    
    # Guardar estado actual para la próxima comparación
    st.session_state.ensemble_previous_state = current_config.copy()
    
    # --- ENTRENAMIENTO DEL MODELO ---
    train_button = st.button("Entrenar y Comparar Modelos", type="primary", key="train_ensemble_models")
    
    if train_button:
        results = train_models(models_config, X_train, y_train, X_test, y_test, random_state, use_regression_model)
        if results:
            st.session_state.ensemble_results = results
            st.session_state.ensemble_y_test = y_test
            st.session_state.use_regression_model = use_regression_model
            st.session_state.ensemble_trained_models = {name: result["model"] for name, result in results.items()}
            st.session_state.ensemble_X_columns = X.columns.tolist()
            st.session_state.ensemble_target_col = target_col
            st.session_state.ensemble_selected_features = selected_features
            st.session_state.ensemble_categorical_cols = categorical_cols if 'categorical_cols' in locals() else []
            st.session_state.ensemble_original_df = df
            st.success("✅ Modelos entrenados exitosamente")
        else:
            st.error("❌ No se pudo entrenar ningún modelo. Revisa parámetros y datos.")
    
    # --- COMPARACIÓN DE MODELOS (PRIMERO) ---
    if ('ensemble_results' in st.session_state and 
        st.session_state.ensemble_results is not None):
        st.subheader("Comparación de Modelos")
        display_comparison_results(
            st.session_state.ensemble_results, 
            st.session_state.ensemble_y_test,
            st.session_state.use_regression_model
        )
    
    # --- PREDICCIÓN CON MODELOS ENTRENADOS (DESPUÉS) ---
    if ('ensemble_trained_models' in st.session_state and 
        st.session_state.ensemble_trained_models is not None):
        
        st.subheader("Predicción con Modelos Entrenados")
        
        # Verificar si la configuración sigue siendo compatible
        if config_changed:
            st.warning("⚠️ La configuración ha cambiado. Los resultados de predicción pueden no ser válidos.")
        
        prediction_method = st.radio(
            "Selecciona el método de predicción:",
            ["Ingresar datos manualmente", "Cargar archivo CSV"],
            key="prediction_method_ensemble"
        )
        
        if prediction_method == "Ingresar datos manualmente":
            st.write("**Ingresa los valores para las variables predictoras:**")
            
            # Crear inputs para cada feature - usar las columnas ORIGINALES
            input_data = {}
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(st.session_state.ensemble_selected_features):
                with col1 if i % 2 == 0 else col2:
                    # Verificar si la feature existe en el DataFrame original
                    if feature in st.session_state.ensemble_original_df.columns:
                        if st.session_state.ensemble_original_df[feature].dtype in ['object', 'category']:
                            # Variable categórica
                            unique_vals = st.session_state.ensemble_original_df[feature].dropna().unique()
                            input_val = st.selectbox(
                                f"{feature}:",
                                options=unique_vals,
                                key=f"input_{feature}_ensemble"
                            )
                        else:
                            # Variable numérica
                            min_val = float(st.session_state.ensemble_original_df[feature].min())
                            max_val = float(st.session_state.ensemble_original_df[feature].max())
                            mean_val = float(st.session_state.ensemble_original_df[feature].mean())
                            
                            input_val = st.number_input(
                                f"{feature} (rango: {min_val:.2f} - {max_val:.2f}):",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"input_{feature}_ensemble"
                            )
                        input_data[feature] = input_val
                    else:
                        st.warning(f"⚠️ Columna '{feature}' no encontrada en datos originales")
            
            # Botón para predecir
            if st.button("Realizar Predicción", type="primary", key="manual_predict_ensemble"):
                try:
                    # Convertir input a DataFrame con las columnas originales
                    input_df = pd.DataFrame([input_data])
                    
                    # Aplicar el mismo preprocesamiento que a los datos de entrenamiento
                    processed_input = preprocess_new_data(
                        input_df, 
                        st.session_state.ensemble_selected_features,
                        st.session_state.ensemble_categorical_cols,
                        st.session_state.ensemble_X_columns
                    )
                    
                    # Realizar predicciones con todos los modelos
                    predictions = {}
                    for model_name, model in st.session_state.ensemble_trained_models.items():
                        if st.session_state.use_regression_model:
                            prediction = model.predict(processed_input)
                            predictions[model_name] = prediction[0]
                        else:
                            prediction = model.predict(processed_input)
                            prediction_proba = model.predict_proba(processed_input) if hasattr(model, "predict_proba") else None
                            predictions[model_name] = {
                                'class': prediction[0],
                                'probabilities': prediction_proba[0] if prediction_proba is not None else None,
                                'classes': model.classes_ if hasattr(model, 'classes_') else None
                            }
                    
                    # Mostrar resultados
                    st.subheader("Resultados de Predicción")
                    
                    if st.session_state.use_regression_model:
                        # Para regresión
                        results_df = pd.DataFrame({
                            'Modelo': list(predictions.keys()),
                            'Predicción': list(predictions.values())
                        })
                        st.dataframe(results_df)
                        
                        # Gráfico de comparación
                        fig, ax = plt.subplots(figsize=(10, 6))
                        models = list(predictions.keys())
                        pred_values = list(predictions.values())
                        
                        bars = ax.bar(models, pred_values, color='skyblue', alpha=0.7, edgecolor='black')
                        ax.set_ylabel(f'Predicción de {st.session_state.ensemble_target_col}')
                        ax.set_title('Comparación de Predicciones entre Modelos')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Añadir valores en las barras
                        for bar, v in zip(bars, pred_values):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                    f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    else:
                        # Para clasificación
                        results_data = []
                        for model_name, pred_info in predictions.items():
                            results_data.append({
                                'Modelo': model_name,
                                'Clase Predicha': pred_info['class'],
                                'Probabilidad Máxima': (max(pred_info['probabilities']) 
                                                       if pred_info['probabilities'] is not None else 'N/A')
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.dataframe(results_df)
                        
                        # Mostrar probabilidades detalladas si están disponibles
                        st.write("**Probabilidades por Clase:**")
                        for model_name, pred_info in predictions.items():
                            if pred_info['probabilities'] is not None and pred_info['classes'] is not None:
                                with st.expander(f"Probabilidades - {model_name}"):
                                    prob_df = pd.DataFrame({
                                        'Clase': pred_info['classes'],
                                        'Probabilidad': pred_info['probabilities']
                                    }).sort_values('Probabilidad', ascending=False)
                                    
                                    st.dataframe(prob_df)
                                    
                                    # Gráfico de probabilidades
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    bars = ax.bar(prob_df['Clase'].astype(str), prob_df['Probabilidad'], 
                                                color='lightcoral', alpha=0.7, edgecolor='black')
                                    ax.set_ylabel('Probabilidad')
                                    ax.set_title(f'Probabilidades de Predicción - {model_name}')
                                    ax.set_ylim(0, 1)
                                    ax.tick_params(axis='x', rotation=45)
                                    
                                    # Añadir valores en las barras
                                    for bar, v in zip(bars, prob_df['Probabilidad']):
                                        height = bar.get_height()
                                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                                f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")
        
        else:  # Cargar archivo CSV
            # Información para el usuario
            with st.expander("ℹ️ Instrucciones para archivo CSV"):
                st.markdown("""
                **⚠️ Formato requerido para archivo CSV de predicción:**
                
                - **Columnas requeridas:** Mismas variables predictoras ORIGINALES usadas en el entrenamiento
                - **Variables predictoras:** {}
                - **Formato de datos:** 
                    - Numéricos: Valores decimales o enteros
                    - Categóricos: Texto (debe coincidir con categorías del entrenamiento)
                - **Codificación:** UTF-8
                - **Separador:** Coma (,)
                - **No incluir:** variable objetivo '{}'
                - **Incluir:** Los nombres de columnas que deben ser idénticos a los originales
                """.format(', '.join(st.session_state.ensemble_selected_features), st.session_state.ensemble_target_col))
            
            st.write("**Carga un archivo CSV con datos para predecir:**")
            
            uploaded_file = st.file_uploader("Selecciona archivo CSV", type=['csv'], 
                                        key="prediction_file_ensemble")
            
            if uploaded_file is not None:
                try:
                    # Cargar datos
                    prediction_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Vista previa de los datos cargados:**")
                    st.dataframe(prediction_df.head(), use_container_width=True)
                    
                    # Verificar columnas requeridas - usar las columnas ORIGINALES
                    required_columns = set(st.session_state.ensemble_selected_features)
                    missing_columns = required_columns - set(prediction_df.columns)
                    
                    if missing_columns:
                        st.error(f"❌ Faltan las siguientes columnas en el archivo: {', '.join(missing_columns)}")
                        st.info("**Columnas requeridas:**")
                        st.write(f"- **Variables predictoras:** {', '.join(st.session_state.ensemble_selected_features)}")
                        st.write(f"- **Formato esperado:** Mismas columnas originales usadas para entrenar el modelo")
                    else:
                        st.success("✅ Todas las columnas requeridas están presentes")
                        
                        # Procesar datos
                        X_pred = prediction_df[st.session_state.ensemble_selected_features]
                        X_pred_processed = preprocess_new_data(
                            X_pred, 
                            st.session_state.ensemble_selected_features,
                            st.session_state.ensemble_categorical_cols,
                            st.session_state.ensemble_X_columns
                        )
                        
                        # Realizar predicciones por lote
                        if st.button("Realizar Predicciones por Lote", type="primary", key="batch_predict_ensemble"):
                            try:
                                with st.spinner("Realizando predicciones..."):
                                    all_predictions = {}
                                    
                                    for model_name, model in st.session_state.ensemble_trained_models.items():
                                        if st.session_state.use_regression_model:
                                            predictions = model.predict(X_pred_processed)
                                            all_predictions[model_name] = predictions
                                        else:
                                            predictions = model.predict(X_pred_processed)
                                            all_predictions[model_name] = predictions
                                    
                                    # Crear DataFrame de resultados
                                    results_df = prediction_df.copy()
                                    for model_name, predictions in all_predictions.items():
                                        results_df[f'PREDICCION_{model_name}'] = predictions
                                    
                                    st.success(f"✅ Predicciones completadas para {len(predictions)} registros")
                                    
                                    # Mostrar resultados
                                    st.write("**Resultados de Predicción:**")
                                    st.dataframe(results_df, use_container_width=True)
                                    
                                    # Descargar resultados
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Descargar Resultados como CSV",
                                        data=csv,
                                        file_name=f"predicciones_ensemble_{st.session_state.ensemble_target_col}.csv",
                                        mime="text/csv",
                                        key="download_results_ensemble"
                                    )
                                    
                            except Exception as e:
                                st.error(f"❌ Error en predicciones por lote: {str(e)}")
                                
                except Exception as e:
                    st.error(f"❌ Error al cargar el archivo: {str(e)}")

def preprocess_new_data(new_data, selected_features, categorical_cols, X_columns):
    """Preprocesa nuevos datos de la misma manera que los datos de entrenamiento"""
    # Aplicar mismo preprocesamiento
    # 1. One-hot encoding para variables categóricas si existen
    if categorical_cols:
        try:
            new_data_cat = pd.get_dummies(new_data[categorical_cols], drop_first=True, dummy_na=False)
            new_data_non_cat = new_data[[col for col in new_data.columns if col not in categorical_cols]]
            new_data_processed = pd.concat([new_data_non_cat, new_data_cat], axis=1)
            
            # Asegurar que tenga las mismas columnas que X (el modelo entrenado)
            missing_cols = set(X_columns) - set(new_data_processed.columns)
            for col in missing_cols:
                new_data_processed[col] = 0
            
            # Reordenar columnas para que coincidan con X
            new_data_processed = new_data_processed[X_columns]
                
        except Exception as e:
            st.error(f"❌ Error en preprocesamiento: {str(e)}")
            # Si falla, intentar con las columnas originales
            new_data_processed = new_data[X_columns]
    else:
        new_data_processed = new_data[X_columns]
    
    return new_data_processed

def configure_random_forest(use_regression_model=False):
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de árboles:", 10, 200, 100, key="rf_n_estimators")
        max_depth = st.slider("Profundidad máxima:", 1, 20, None, key="rf_max_depth")
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2, key="rf_min_samples")
        
        # CRITERIOS DIFERENTES PARA REGRESIÓN VS CLASIFICACIÓN
        if use_regression_model:
            criterion = st.selectbox("Criterio:", ["squared_error", "absolute_error", "friedman_mse"], 
                                   key="rf_criterion_reg")
        else:
            criterion = st.selectbox("Criterio:", ["gini", "entropy"], key="rf_criterion_clf")
    
    return {"n_estimators": n_estimators, "max_depth": max_depth,
            "min_samples_split": min_samples_split, "criterion": criterion}

def configure_adaboost(use_regression_model=False):
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 50, key="ab_n_estimators")
    with col2:
        learning_rate = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01, key="ab_learning_rate")
    
    # PARÁMETROS ESPECÍFICOS PARA REGRESIÓN
    if use_regression_model:
        col3, col4 = st.columns(2)
        with col3:
            loss = st.selectbox("Función de pérdida:", ["linear", "square", "exponential"], 
                              key="ab_loss_reg")
        return {"n_estimators": n_estimators, "learning_rate": learning_rate, "loss": loss}
    else:
        return {"n_estimators": n_estimators, "learning_rate": learning_rate}

def configure_gradient_boosting(use_regression_model=False):
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 200, 100, key="gb_n_estimators")
        learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1, 0.01, key="gb_learning_rate")
    with col2:
        max_depth = st.slider("Profundidad máxima:", 1, 10, 3, key="gb_max_depth")
        min_samples_split = st.slider("Mínimo samples split:", 2, 20, 2, key="gb_min_samples")
    
    # CRITERIOS DIFERENTES PARA REGRESIÓN
    if use_regression_model:
        criterion = st.selectbox("Criterio:", ["friedman_mse", "squared_error"], 
                               key="gb_criterion_reg")
        loss = st.selectbox("Función de pérdida:", ["squared_error", "absolute_error", "huber", "quantile"], 
                          key="gb_loss_reg")
        return {"n_estimators": n_estimators, "learning_rate": learning_rate,
                "max_depth": max_depth, "min_samples_split": min_samples_split,
                "criterion": criterion, "loss": loss}
    else:
        return {"n_estimators": n_estimators, "learning_rate": learning_rate,
                "max_depth": max_depth, "min_samples_split": min_samples_split}

def configure_bagging():
    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Número de estimadores:", 10, 100, 10, key="bag_n_estimators")
    with col2:
        max_samples = st.slider("Máximo samples:", 0.1, 1.0, 1.0, 0.1, key="bag_max_samples")
    return {"n_estimators": n_estimators, "max_samples": max_samples}

def train_models(models_config, X_train, y_train, X_test, y_test, random_state, use_regression_model=False):
    results = {}
    for model_name, config in models_config.items():
        with st.spinner(f"Entrenando {model_name}..."):
            try:
                # Seleccionar el modelo adecuado según el tipo de variable objetivo
                if use_regression_model:
                    # Modelos de regresión para variables numéricas
                    if model_name == "Random Forest":
                        # Validar criterio para regresión
                        if config.get("criterion") in ["gini", "entropy"]:
                            config["criterion"] = "squared_error"  # Valor por defecto para regresión
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
                if use_regression_model:
                    results[model_name] = {
                        "y_pred": y_pred,
                        "model": model
                    }
                else:
                    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    results[model_name] = {
                        "y_pred": y_pred,
                        "y_prob": y_prob,
                        "classes": model.classes_,
                        "model": model
                    }
                    
                st.success(f"✅ {model_name} entrenado exitosamente")

            except Exception as e:
                st.error(f"❌ Error al entrenar {model_name}: {str(e)}")
                # Debug information
                st.error(f"Configuración usada: {config}")
    
    return results

def display_comparison_results(results, y_test, use_regression_model):    
    comparison_data = []
    
    if use_regression_model:
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
            weighted_avg = report.get('weighted avg', {})
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
            if use_regression_model:
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
                    use_regression_model
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

def show_model_results(y_test, y_pred, y_prob, classes, model_name, use_regression_model):
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