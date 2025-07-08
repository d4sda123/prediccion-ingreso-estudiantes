# Guía de Uso de Modelos Guardados

Este documento explica cómo usar los modelos de machine learning entrenados y guardados para hacer predicciones.

## 📁 Estructura de Archivos

Después de ejecutar `train.py`, se creará un directorio `models/` con los siguientes archivos:

```
models/
├── regresion_lineal.pkl              # Modelo de Regresión Lineal
├── bosques_aleatorios.pkl            # Modelo de Bosques Aleatorios
├── regresion_de_vectores_de_soporte.pkl  # Modelo SVR
├── potenciacion_de_gradiente.pkl     # Modelo de Gradient Boosting
├── best_model.pkl                    # El mejor modelo (copia del mejor)
├── standard_scaler.pkl               # StandardScaler para normalización
├── label_encoders.pkl                # LabelEncoders para variables categóricas
└── column_info.pkl                   # Información de columnas y estructura
```

## 🚀 Uso Rápido

### 1. Cargar Modelos

```python
from load_models import load_models_and_scalers

# Cargar todos los modelos y scalers
models_dict = load_models_and_scalers()

# Acceder a componentes específicos
models = models_dict['models']           # Todos los modelos
best_model = models_dict['best_model']   # El mejor modelo
scaler = models_dict['scaler']           # StandardScaler
label_encoders = models_dict['label_encoders']  # LabelEncoders
column_info = models_dict['column_info'] # Información de columnas
```

### 2. Preprocesar Nuevos Datos

```python
from load_models import preprocess_new_data

# Supongamos que tienes nuevos datos
new_data = pd.DataFrame({
    'COLEGIO_ANIO_EGRESO': [2023],
    'ANIO_POSTULA': [2024],
    'CICLO_POSTULA': [1],
    'ANIO_NACIMIENTO': [1995],
    'CALIF_FINAL': [15.5],
    'COLEGIO': ['LA DIVINA PROVIDENCIA'],
    'ESPECIALIDAD': ['INGENIERÍA DE SISTEMAS'],
    'SEXO': ['MASCULINO'],
    'MODALIDAD': ['ORDINARIO'],
    'INGRESO': ['SI']
})

# Preprocesar los datos
processed_data = preprocess_new_data(
    new_data, 
    models_dict['label_encoders'], 
    models_dict['scaler'], 
    models_dict['column_info']
)
```

### 3. Hacer Predicciones

```python
from load_models import predict_with_models

# Predicciones con todos los modelos
predictions = predict_with_models(processed_data, models_dict)

# O usar un modelo específico
best_model = models_dict['best_model']
prediction = best_model.predict(processed_data)
```

## 📊 Ejemplos de Uso

### Ejemplo Completo
Ejecuta el archivo `example_prediction.py` para ver un ejemplo completo:

```bash
python example_prediction.py
```

### Ejemplo Rápido
Para el uso más simple, ejecuta `quick_start.py`:

```bash
python quick_start.py
```

```bash
python example_prediction.py
```

Este script demuestra:
- Carga de modelos
- Creación de datos de ejemplo
- Preprocesamiento
- Predicciones con todos los modelos
- Análisis de resultados

## 🔧 Funciones Disponibles

### `load_models_and_scalers()`
Carga todos los modelos y componentes de preprocesamiento guardados.

**Retorna:** Diccionario con:
- `models`: Todos los modelos entrenados
- `best_model`: El mejor modelo
- `scaler`: StandardScaler
- `label_encoders`: LabelEncoders para variables categóricas
- `column_info`: Información de estructura de datos

### `preprocess_new_data(data, label_encoders, scaler, column_info)`
Preprocesa nuevos datos usando los encoders y scaler guardados.

**Parámetros:**
- `data`: DataFrame con nuevos datos
- `label_encoders`: LabelEncoders cargados
- `scaler`: StandardScaler cargado
- `column_info`: Información de columnas

**Retorna:** Array numpy con datos preprocesados

### `predict_with_models(new_data, models_dict)`
Realiza predicciones con todos los modelos cargados.

**Parámetros:**
- `new_data`: Datos preprocesados
- `models_dict`: Diccionario con modelos cargados

**Retorna:** Diccionario con predicciones de cada modelo

## 📋 Requisitos de Datos

Los nuevos datos deben tener las siguientes columnas:

### Columnas Categóricas:
- `COLEGIO`: Nombre del colegio
- `ESPECIALIDAD`: Especialidad de estudio
- `SEXO`: Género (MASCULINO/FEMENINO)
- `MODALIDAD`: Modalidad de ingreso (ORDINARIO, etc.)
- `INGRESO`: Estado de ingreso (SI/NO)

### Columnas Numéricas:
- `COLEGIO_ANIO_EGRESO`: Año de egreso del colegio
- `ANIO_POSTULA`: Año de postulación
- `CICLO_POSTULA`: Ciclo de postulación
- `ANIO_NACIMIENTO`: Año de nacimiento
- `CALIF_FINAL`: Calificación final (característica más importante)

## ⚠️ Notas Importantes

1. **Formato de Datos**: Los nuevos datos deben tener exactamente las mismas columnas que se usaron durante el entrenamiento.

2. **Valores Categóricos**: Los valores en las columnas categóricas deben estar dentro del rango de valores vistos durante el entrenamiento.

3. **Escalado**: Algunos modelos (Regresión Lineal, SVR) usan datos escalados, mientras que otros (Bosques Aleatorios, Gradient Boosting) no.

4. **Predicciones**: Las predicciones son valores continuos entre 0 y 1. Para clasificación binaria, usa un umbral de 0.5.

## 🛠️ Solución de Problemas

### Error: "El directorio 'models' no existe"
- Ejecuta `train.py` primero para entrenar y guardar los modelos.

### Error: "ValueError: Found input variables with inconsistent numbers of features"
- Verifica que los nuevos datos tengan las mismas columnas que se usaron durante el entrenamiento.

### Error: "ValueError: y contains previously unseen labels"
- Los valores categóricos en los nuevos datos deben estar dentro del rango de valores del entrenamiento.

## 📈 Interpretación de Resultados

- **Predicción > 0.5**: Probable INGRESO
- **Predicción ≤ 0.5**: Probable NO INGRESO
- **Valor más alto**: Mayor probabilidad de ingreso
- **Valor más bajo**: Menor probabilidad de ingreso

## 🔄 Actualización de Modelos

Para actualizar los modelos con nuevos datos:
1. Agrega los nuevos datos al dataset original
2. Ejecuta `train.py` nuevamente
3. Los modelos se sobrescribirán automáticamente 