import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
import graphviz
from io import StringIO
import pydotplus
from itertools import cycle

def decision_tree_module(df):
    st.subheader("Configuración del Árbol de Decisión")
    
    # --- INICIALIZACIÓN DE ESTADO ---
    if 'dt_current_state' not in st.session_state:
        st.session_state.dt_current_state = {}
    
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
        'criterion': None,
        'max_depth': None,
        'min_samples_split': None,
        'min_samples_leaf': None
    }
    
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
        current_config['target_col'] = target_col
    
    with col2:
        # Selección de variables predictoras
        available_features = [col for col in df.columns if col != target_col]
        selected_features = st.multiselect(
            "Selecciona las variables predictoras:",
            available_features,
            default=available_features,
            key="features_select_dt"
        )
        current_config['selected_features'] = tuple(selected_features)  # Convertir a tuple para hash
    
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
            key="cleaning_strategy_dt"
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
            key="numeric_option_dt"
        )
        current_config['numeric_option'] = numeric_option
        
        if numeric_option == "Discretizar en intervalos para clasificación":
            use_regression_model = False
            # Opciones de discretización
            discretize_method = st.selectbox(
                "Método de discretización:",
                ["Intervalos iguales", "Cuantiles", "Personalizado"],
                key="discretize_method_dt"
            )
            current_config['discretize_method'] = discretize_method
            
            if discretize_method == "Intervalos iguales":
                n_bins = st.slider("Número de intervalos:", 2, 10, 4, key="n_bins_equal_dt")
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
                n_bins = st.slider("Número de cuantiles:", 2, 10, 4, key="n_bins_quantile_dt")
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
                    key="custom_cutpoints_dt"
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
            st.info("ℹ️ Se utilizará un modelo de regresión (DecisionTreeRegressor) para la variable objetivo numérica")
        
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
                            key="test_size_dt")
        current_config['test_size'] = test_size
    
    with col2:
        random_state = st.number_input("Random state:", 0, 100, 42, key="random_state_dt")
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
    
    # --- Configuración de hiperparámetros según selección ---
    st.subheader("Configuración de Hiperparámetros")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if use_regression_model:
            # Hiperparámetros para regresión
            criterion = st.selectbox(
                "Criterio de división:",
                ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                help="Criterios específicos para modelos de regresión",
                key="criterion_reg_dt"
            )
        else:
            # Hiperparámetros para clasificación
            criterion = st.selectbox(
                "Criterio de división:",
                ["gini", "entropy"],
                help="GINI: medida de impureza | Entropy: ganancia de información",
                key="criterion_clf_dt"
            )
        current_config['criterion'] = criterion
        
        max_depth = st.slider("Profundidad máxima:", 1, 20, 5, key="max_depth_dt")
        current_config['max_depth'] = max_depth
    
    with col2:
        min_samples_split = st.slider("Mínimo samples para split:", 2, 20, 2, 
                                    key="min_samples_split_dt")
        current_config['min_samples_split'] = min_samples_split
        
        min_samples_leaf = st.slider("Mínimo samples por hoja:", 1, 10, 1, 
                                   key="min_samples_leaf_dt")
        current_config['min_samples_leaf'] = min_samples_leaf
    
    # --- DETECCIÓN DE CAMBIOS Y LIMPIEZA DE RESULTADOS ---
    config_changed = False
    if hasattr(st.session_state, 'dt_previous_state'):
        # Comparar configuración actual con anterior
        for key in current_config:
            if current_config[key] != st.session_state.dt_previous_state.get(key):
                config_changed = True
                break
    
    # Si la configuración cambió, limpiar resultados anteriores
    if config_changed:
        st.session_state.model_trained = False
        st.session_state.trained_model = None
        st.info("🔄 Configuración modificada. Entrena el modelo nuevamente.")
    
    # Guardar estado actual para la próxima comparación
    st.session_state.dt_previous_state = current_config.copy()
    
    # --- ENTRENAMIENTO DEL MODELO ---
    train_button = st.button("Entrenar Árbol de Decisión", type="primary", key="train_model_dt")
    
    if train_button:
        try:
            with st.spinner("Entrenando modelo..."):
                # Usar regressor o classifier según el tipo de variable objetivo
                if use_regression_model:
                    # Para regresión
                    model = DecisionTreeRegressor(
                        criterion=criterion,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=random_state
                    )
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_prob = None  # No hay probabilidades en regresión
                    
                    st.success("✅ Modelo de regresión entrenado exitosamente")
                    display_regression_results(y_test, y_pred)
                else:
                    # Para clasificación, verificamos que haya suficientes clases
                    if len(np.unique(y_train)) < 2:
                        st.error("❌ El conjunto de entrenamiento debe tener al menos 2 clases diferentes")
                        return
                    
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
                    
                    st.success("✅ Modelo de clasificación entrenado exitosamente")
                    display_classification_results(y_test, y_pred, y_prob, model.classes_, use_regression_model)
                
                # Visualización del árbol
                st.subheader("Visualización del Árbol de Decisión")
                
                # Crear visualización del árbol
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Determinar feature_names y class_names
                feature_names = X.columns.tolist()
                
                if use_regression_model:
                    class_names = None  # No hay nombres de clase para regresión
                else:
                    if hasattr(model, 'classes_'):
                        class_names = [str(c) for c in model.classes_]
                    else:
                        class_names = [str(c) for c in np.unique(y)]
                
                # Dibujar el árbol con configuración predeterminada
                plot_tree(model, 
                        max_depth=3,  # Profundidad fija para mejor visualización
                        feature_names=feature_names,
                        class_names=class_names,
                        filled=True,
                        rounded=True,
                        ax=ax,
                        fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Guardar el modelo en session_state para usarlo en predicciones
                st.session_state.trained_model = model
                st.session_state.model_trained = True
                st.session_state.use_regression_model = use_regression_model
                st.session_state.target_col = target_col
                st.session_state.selected_features = selected_features
                st.session_state.X_columns = X.columns.tolist()
                st.session_state.categorical_cols = categorical_cols if 'categorical_cols' in locals() else []
                st.session_state.original_df = df
        
        except Exception as e:
            st.error(f"❌ Error al entrenar el modelo: {str(e)}")
            st.write("Sugerencia: Verifica la variable objetivo, ajusta los hiperparámetros o revisa las variables predictoras.")

    # --- PREDICCIÓN CON EL MODELO ENTRENADO ---
    # Solo mostrar si el modelo ha sido entrenado
    if 'model_trained' in st.session_state and st.session_state.model_trained:
        st.subheader("Predicción con el Modelo Entrenado")
        
        # Verificar si la configuración sigue siendo compatible
        if config_changed:
            st.warning("⚠️ La configuración ha cambiado. Los resultados de predicción pueden no ser válidos.")
        
        # Recuperar variables del session_state
        model = st.session_state.trained_model
        use_regression_model = st.session_state.use_regression_model
        target_col = st.session_state.target_col
        selected_features = st.session_state.selected_features
        X_columns = st.session_state.X_columns
        categorical_cols = st.session_state.categorical_cols
        original_df = st.session_state.original_df
        
        prediction_method = st.radio(
            "Selecciona el método de predicción:",
            ["Ingresar datos manualmente", "Cargar archivo CSV"],
            key="prediction_method_dt"
        )

        if prediction_method == "Ingresar datos manualmente":
            st.write("**Ingresa los valores para las variables predictoras:**")
            
            # Crear inputs para cada feature - usar las columnas ORIGINALES
            input_data = {}
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(selected_features):
                with col1 if i % 2 == 0 else col2:
                    # Verificar si la feature existe en el DataFrame original
                    if feature in original_df.columns:
                        if original_df[feature].dtype in ['object', 'category']:
                            # Variable categórica
                            unique_vals = original_df[feature].dropna().unique()
                            input_val = st.selectbox(
                                f"{feature}:",
                                options=unique_vals,
                                key=f"input_{feature}_dt"
                            )
                        else:
                            # Variable numérica
                            min_val = float(original_df[feature].min())
                            max_val = float(original_df[feature].max())
                            mean_val = float(original_df[feature].mean())
                            
                            input_val = st.number_input(
                                f"{feature} (rango: {min_val:.2f} - {max_val:.2f}):",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"input_{feature}_dt"
                            )
                        input_data[feature] = input_val
                    else:
                        st.warning(f"⚠️ Columna '{feature}' no encontrada en datos originales")
            
            # Botón para predecir
            if st.button("Realizar Predicción", type="primary", key="manual_predict_dt"):
                try:
                    # Convertir input a DataFrame con las columnas originales
                    input_df = pd.DataFrame([input_data])
                    
                    # Aplicar el mismo preprocesamiento que a los datos de entrenamiento
                    # 1. One-hot encoding para variables categóricas si existen
                    if categorical_cols:
                        # Crear dummies para las columnas categóricas
                        input_cat = pd.get_dummies(input_df[categorical_cols], drop_first=True, dummy_na=False)
                        
                        # Mantener las columnas numéricas
                        input_non_cat = input_df[[col for col in input_df.columns if col not in categorical_cols]]
                        
                        # Combinar ambos conjuntos de datos
                        input_processed = pd.concat([input_non_cat, input_cat], axis=1)
                        
                        # Asegurar que tenga las mismas columnas que X (el modelo entrenado)
                        missing_cols = set(X_columns) - set(input_processed.columns)
                        for col in missing_cols:
                            input_processed[col] = 0
                        
                        # Reordenar columnas para que coincidan con X
                        input_processed = input_processed[X_columns]
                            
                    else:
                        # Si no hay variables categóricas, usar directamente
                        input_processed = input_df[X_columns]
                    
                    # Realizar predicción
                    if use_regression_model:
                        prediction = model.predict(input_processed)
                        st.success(f"**📊 Predicción:** {prediction[0]:.4f}")
                        
                        # Mostrar interpretación para regresión
                        st.info(f"El modelo predice un valor de **{prediction[0]:.4f}** para la variable objetivo '{target_col}'")
                        
                    else:
                        prediction = model.predict(input_processed)
                        prediction_proba = model.predict_proba(input_processed)
                        
                        st.success(f"**📊 Predicción:** {prediction[0]}")
                        
                        # Mostrar probabilidades
                        st.write("**Probabilidades por clase:**")
                        prob_df = pd.DataFrame({
                            'Clase': model.classes_,
                            'Probabilidad': prediction_proba[0]
                        }).sort_values('Probabilidad', ascending=False)
                        
                        # Gráfico de probabilidades
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.bar(prob_df['Clase'].astype(str), prob_df['Probabilidad'], 
                                    color='skyblue', edgecolor='black', alpha=0.7)
                        ax.set_ylabel('Probabilidad')
                        ax.set_xlabel('Clases')
                        ax.set_title('Probabilidades de Predicción por Clase')
                        ax.set_ylim(0, 1)
                        
                        # Añadir valores en las barras
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {str(e)}")
                    st.info("ℹ️ Esto puede ocurrir si hay discrepancias en el preprocesamiento de datos")

        else:  
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
                """.format(', '.join(selected_features), target_col))
                # Cargar archivo CSV
            st.write("**Carga un archivo CSV con datos para predecir:**")
            
            uploaded_file = st.file_uploader("Selecciona archivo CSV", type=['csv'], 
                                        key="prediction_file_dt")
            
            if uploaded_file is not None:
                try:
                    # Cargar datos
                    prediction_df = pd.read_csv(uploaded_file)
                    
                    st.write("**Vista previa de los datos cargados:**")
                    st.dataframe(prediction_df.head(), use_container_width=True)
                    
                    # Verificar columnas requeridas - usar las columnas ORIGINALES
                    required_columns = set(selected_features)
                    missing_columns = required_columns - set(prediction_df.columns)
                    
                    if missing_columns:
                        st.error(f"❌ Faltan las siguientes columnas en el archivo: {', '.join(missing_columns)}")
                        st.info("**Columnas requeridas:**")
                        st.write(f"- **Variables predictoras:** {', '.join(selected_features)}")
                        st.write(f"- **Formato esperado:** Mismas columnas originales usadas para entrenar el modelo")
                    else:
                        st.success("✅ Todas las columnas requeridas están presentes")
                        
                        # Procesar datos igual que en entrenamiento
                        X_pred = prediction_df[selected_features]
                        
                        # Aplicar mismo preprocesamiento
                        # 1. Manejar infinitos
                        inf_mask = np.any(np.isinf(X_pred.select_dtypes(include=['float64', 'int64']).values), axis=1)
                        if inf_mask.any():
                            st.warning(f"⚠️ Se detectaron {inf_mask.sum()} filas con valores infinitos. Serán eliminadas.")
                            X_pred = X_pred[~inf_mask]
                        
                        # 2. One-hot encoding si hay variables categóricas
                        if categorical_cols:
                            try:
                                X_pred_cat = pd.get_dummies(X_pred[categorical_cols], drop_first=True, dummy_na=False)
                                X_pred_non_cat = X_pred[[col for col in X_pred.columns if col not in categorical_cols]]
                                X_pred_processed = pd.concat([X_pred_non_cat, X_pred_cat], axis=1)
                                
                                # Asegurar mismas columnas que el modelo
                                missing_cols = set(X_columns) - set(X_pred_processed.columns)
                                for col in missing_cols:
                                    X_pred_processed[col] = 0
                                
                                # Reordenar columnas
                                X_pred_processed = X_pred_processed[X_columns]
                                    
                            except Exception as e:
                                st.error(f"❌ Error en preprocesamiento: {str(e)}")
                                st.info("Intentando procesamiento alternativo...")
                                X_pred_processed = X_pred[X_columns]  # Usar columnas originales
                        else:
                            X_pred_processed = X_pred[X_columns]
                        
                        # Realizar predicciones
                        if st.button("Realizar Predicciones por Lote", type="primary", key="batch_predict_dt"):
                            try:
                                with st.spinner("Realizando predicciones..."):
                                    if use_regression_model:
                                        predictions = model.predict(X_pred_processed)
                                        results_df = prediction_df.copy()
                                        results_df[f'PREDICCION_{target_col}'] = predictions
                                        
                                        st.success(f"✅ Predicciones completadas para {len(predictions)} registros")
                                        
                                        # Mostrar resultados
                                        st.write("**Resultados de Predicción:**")
                                        st.dataframe(results_df, use_container_width=True)                                   
                                        
                                    else:
                                        predictions = model.predict(X_pred_processed)
                                        predictions_proba = model.predict_proba(X_pred_processed)
                                        
                                        results_df = prediction_df.copy()
                                        results_df[f'PREDICCION_{target_col}'] = predictions
                                        results_df['PROBABILIDAD_MAXIMA'] = predictions_proba.max(axis=1)
                                        
                                        # Añadir probabilidades por clase
                                        for i, class_name in enumerate(model.classes_):
                                            results_df[f'PROB_{class_name}'] = predictions_proba[:, i]
                                        
                                        st.success(f"✅ Predicciones completadas para {len(predictions)} registros")
                                        
                                        # Mostrar resultados
                                        st.write("**Resultados de Predicción:**")
                                        st.dataframe(results_df, use_container_width=True)
                                        
                                        # Distribución de predicciones
                                        st.write("**Distribución de Predicciones:**")
                                        pred_counts = results_df[f'PREDICCION_{target_col}'].value_counts()
                                        
                                        # Gráfico de distribución de clases predichas
                                        fig, ax = plt.subplots(figsize=(10, 6))

                                        # Gráfico de distribución
                                        ax.bar(pred_counts.index.astype(str), pred_counts.values, 
                                            color='lightcoral', alpha=0.7, edgecolor='black')
                                        ax.set_xlabel('Clase Predicha')
                                        ax.set_ylabel('Cantidad')
                                        ax.set_title('Distribución de Clases Predichas')
                                        ax.tick_params(axis='x', rotation=45)

                                        # Añadir valores en las barras
                                        for i, v in enumerate(pred_counts.values):
                                            ax.text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
                                        
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                
                                # Descargar resultados
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Resultados como CSV",
                                    data=csv,
                                    file_name=f"predicciones_arbol_decision_{target_col}.csv",
                                    mime="text/csv",
                                    key="download_results_dt"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error en predicciones por lote: {str(e)}")
                                
                except Exception as e:
                    st.error(f"❌ Error al cargar el archivo: {str(e)}")

# [Las funciones display_regression_results, display_classification_results, e interpretar_auc permanecen iguales]
def display_regression_results(y_test, y_pred):
    """Muestra los resultados para modelos de regresión"""
    st.subheader("Resultados de la Evaluación (Regresión)")
    
    # Calcular métricas de regresión
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Mostrar métricas en una tabla
    metrics_df = pd.DataFrame({
        'Métrica': ['Error Cuadrático Medio (MSE)', 'Raíz del Error Cuadrático Medio (RMSE)', 
                   'Error Absoluto Medio (MAE)', 'Coeficiente de Determinación (R²)'],
        'Valor': [mse, rmse, mae, r2]
    })
    
    st.write("**Métricas de Evaluación:**")
    st.dataframe(metrics_df)
    
    # Visualización de predicciones vs valores reales
    st.write("**Predicciones vs Valores Reales:**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    
    # Añadir línea de referencia perfecta
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Predicciones')
    ax.set_title('Comparación de Predicciones vs Valores Reales')
    
    # Añadir texto con métricas
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    st.pyplot(fig)
    
    # Histograma de residuos
    residuals = y_test - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlabel('Residuos')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Residuos')
    
    st.pyplot(fig)

def display_classification_results(y_test, y_pred, y_prob, classes, use_regression_model):
    """Muestra los resultados para modelos de clasificación"""
    st.subheader("Resultados de la Evaluación (Clasificación)")
    
    # Determinar el tipo de problema
    n_classes = len(classes)
    is_binary_classification = n_classes == 2
    is_multiclass_classification = n_classes > 2
    
    # Mostrar curvas ROC solo para clasificación (no regresión)
    if not use_regression_model and y_prob is not None:
        if is_binary_classification:
            tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curva ROC"])
        elif is_multiclass_classification:
            tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación", "Curvas ROC"])
        else:
            tab1, tab2 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación"])
    else:
        tab1, tab2 = st.tabs(["Matriz de Confusión", "Reporte de Clasificación"])
    
    # Matriz de Confusión
    with tab1:
        st.write("**Matriz de Confusión:**")
        cm = confusion_matrix(y_test, y_pred)

        num_classes = len(classes)

        # Ajustes dinámicos para matriz de confusión
        if num_classes == 2:
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
            square=True
        )

        # Mejorar etiquetas y título
        ax.set_xlabel('Predicciones', fontsize=tick_font_size + 2, weight='bold')
        ax.set_ylabel('Valores Reales', fontsize=tick_font_size + 2, weight='bold')
        ax.set_title('Matriz de Confusión', fontsize=font_size + 4, weight='bold', pad=20)

        # Rotar etiquetas horizontales para evitar traslape
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_font_size)
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
            
    # Curvas ROC solo para clasificación (no regresión) y con soporte multiclase
    if not use_regression_model and y_prob is not None:
        if is_binary_classification:
            with tab3:
                with st.expander("Curva ROC (Receiver Operating Characteristic)"):
                    st.markdown("""
                La **Curva ROC** es una representación gráfica que muestra la capacidad de un clasificador 
                para diferenciar entre clases. Se basa en two métricas:
                
                - **Tasa de falsos positivos (False Positive Rate - FPR):** Proporción de negativos incorrectamente clasificados como positivos.
                - **Tasa de verdaderos positivos (True Positive Rate - TPR):** Proporción de positivos correctamente identificados.
                
                La curva muestra la relación entre TPR y FPR para diferentes umbrales de decisión.
                
                **Nota:** La curva ROC solo está disponible para problemas de clasificación binaria.
                """)
                
                try:
                    # Convertir y_test a numérico
                    class_mapping = {class_name: i for i, class_name in enumerate(classes)}
                    y_test_numeric = np.array([class_mapping[label] for label in y_test])
                    
                    # Gráfico: CURVA ROC
                    st.subheader("Curva ROC")

                    fig_width = 10
                    fig_height = 8
                    fig_roc, ax_roc = plt.subplots(figsize=(fig_width, fig_height))

                    # Clasificación binaria
                    pos_label = 1
                    
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
                            
                    # Mostrar métricas numéricas
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
        
        elif is_multiclass_classification:
            with tab3:
                st.subheader("Curvas ROC para Clasificación Multiclase")
                
                try:
                    # Convertir y_test a formato numérico
                    le = LabelEncoder()
                    y_test_encoded = le.fit_transform(y_test)
                    
                    # Binarizar las etiquetas para multiclase
                    y_test_bin = label_binarize(y_test_encoded, classes=np.unique(y_test_encoded))
                    n_classes = y_test_bin.shape[1]
                    
                    # Calcular curvas ROC y AUC para cada clase
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Calcular micro-average ROC curve y ROC area
                    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
                    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
                    # Calcular macro-average ROC curve y ROC area
                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in range(n_classes):
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                    mean_tpr /= n_classes
                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr
                    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                    
                    # Plot all ROC curves
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 
                                  'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])
                    
                    for i, color in zip(range(n_classes), colors):
                        ax.plot(fpr[i], tpr[i], color=color, lw=2,
                                label='ROC clase {0} (AUC = {1:0.2f})'
                                ''.format(classes[i], roc_auc[i]))
                    
                    # Plot micro and macro averages
                    ax.plot(fpr["micro"], tpr["micro"],
                            label='ROC micro-promedio (AUC = {0:0.2f})'
                            ''.format(roc_auc["micro"]),
                            color='deeppink', linestyle=':', linewidth=4)
                    
                    ax.plot(fpr["macro"], tpr["macro"],
                            label='ROC macro-promedio (AUC = {0:0.2f})'
                            ''.format(roc_auc["macro"]),
                            color='navy', linestyle=':', linewidth=4)
                    
                    # Plot diagonal line
                    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Línea base (AUC = 0.5)')
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
                    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
                    ax.set_title('Curvas ROC - Clasificación Multiclase')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Mostrar métricas de AUC
                    st.subheader("Métricas AUC por Clase")
                    
                    auc_data = []
                    for i in range(n_classes):
                        auc_data.append({
                            'Clase': classes[i],
                            'AUC': roc_auc[i],
                            'Interpretación': interpretar_auc(roc_auc[i])
                        })
                    
                    # Añadir promedios
                    auc_data.append({
                        'Clase': 'Micro-promedio',
                        'AUC': roc_auc["micro"],
                        'Interpretación': interpretar_auc(roc_auc["micro"])
                    })
                    auc_data.append({
                        'Clase': 'Macro-promedio', 
                        'AUC': roc_auc["macro"],
                        'Interpretación': interpretar_auc(roc_auc["macro"])
                    })
                    
                    auc_df = pd.DataFrame(auc_data)
                    st.dataframe(auc_df.style.format({'AUC': '{:.4f}'}))
                    
                    # Gráfico de barras para AUC por clase
                    fig_auc, ax_auc = plt.subplots(figsize=(12, 6))
                    
                    classes_auc = [f'Clase {i}' for i in range(n_classes)] + ['Micro', 'Macro']
                    auc_values = [roc_auc[i] for i in range(n_classes)] + [roc_auc["micro"], roc_auc["macro"]]
                    colors_auc = ['lightblue' if x >= 0.7 else 'lightcoral' for x in auc_values]
                    
                    bars = ax_auc.bar(classes_auc, auc_values, color=colors_auc, edgecolor='black', alpha=0.8)
                    ax_auc.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Aleatorio')
                    ax_auc.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='Aceptable')
                    ax_auc.set_ylim(0, 1.1)
                    ax_auc.set_ylabel('Valor AUC')
                    ax_auc.set_title('Métricas AUC por Clase - Clasificación Multiclase')
                    ax_auc.legend()
                    ax_auc.tick_params(axis='x', rotation=45)
                    
                    # Añadir valores en las barras
                    for bar, v in zip(bars, auc_values):
                        height = bar.get_height()
                        ax_auc.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig_auc)
                    
                except Exception as e:
                    st.error(f"❌ Error al calcular curvas ROC multiclase: {str(e)}")
                    st.info("ℹ️ Esto puede ocurrir cuando hay problemas con las probabilidades predichas o las clases objetivo")

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