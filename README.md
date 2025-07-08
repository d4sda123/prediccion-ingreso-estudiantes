# Predictor de Ingreso Universitario

Este proyecto es una aplicación web desarrollada con Streamlit que predice la probabilidad de ingreso universitario de un estudiante, utilizando modelos de machine learning entrenados con datos históricos de admisión.

## Características
- Interfaz web intuitiva para ingresar datos del estudiante
- Predicción de probabilidad de ingreso basada en modelos de aprendizaje automático (Random Forest, etc.)
- Generación de reportes PDF personalizados con los resultados
- Scripts para entrenamiento, evaluación y comparación de modelos
- Visualización de datos y resultados de modelos

## Instalación
1. **Clona el repositorio:**
   ```bash
   git clone <URL-del-repositorio>
   cd prediccion-ingreso-estudiantes
   ```
2. **Instala las dependencias:**
   Se recomienda el uso de un entorno virtual.
   ```bash
   pip install -r requeriments.txt
   ```

## Requisitos
Las principales dependencias se encuentran en `requeriments.txt`:
- streamlit
- seaborn
- statsmodel
- reportlab

Instálalas con:
```bash
pip install -r requeriments.txt
```

## Uso
### Entrenar modelos
El script `train.py` permite entrenar y comparar diferentes modelos de machine learning usando el dataset de admisión.
```bash
python train.py
```

### Probar el modelo
El script `test.py` muestra cómo cargar el modelo y realizar una predicción con datos de ejemplo.
```bash
python test.py
```

### Ejecutar la aplicación web
```bash
streamlit run app.py
```
Esto abrirá la interfaz en tu navegador, donde podrás ingresar los datos del estudiante y obtener la predicción.