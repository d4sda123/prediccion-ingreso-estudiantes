# Predictor de Ingreso Universitario

Este proyecto es una aplicación web desarrollada con Streamlit que predice la probabilidad de ingreso universitario de un estudiante, utilizando modelos de machine learning entrenados con datos históricos de admisión.

## Características
- Interfaz web intuitiva para ingresar datos del estudiante
- Predicción de probabilidad de ingreso basada en modelos de aprendizaje automático (Random Forest, etc.)
- Generación de reportes PDF personalizados con los resultados
- Scripts para entrenamiento, evaluación y comparación de modelos
- Visualización de datos y resultados de modelos

## Conjunto de datos
Se utilizó el conjunto de datos ["Student Performance Data Set"](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set), que contiene información acerca de estudiantes de escuelas secundarias, como características destacan las horas de estudio, el tiempo libre, el acceso a internet, situación económica, trabajos de los padres, etc.

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

### Ejecutar la aplicación web
```bash
streamlit run app.py
```
Esto abrirá la interfaz en tu navegador, donde podrás ingresar los datos del estudiante y obtener la predicción.

## Despliegue en Cloud
### Streamlit Cloud
- Almacenar el script de streamlit `train.py` en un repositorio github.
- Iniciar sesión en [Streamlit Cloud](https://share.streamlit.io/).
- Seleccionar la opción de crear nueva aplicación.
- Enlazar tu cuenta de Github.
- Seleccionar el repositorio creado.
- Seleccionar la rama y el script.
- Añadir un nombre de URL (Opcional).
- Seleccionar desplegar.