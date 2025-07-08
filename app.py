import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import joblib
import os
from make_pdf import create_pdf, add_title, add_subtitle, add_paragraph, add_spacer, add_table, build_pdf
import tempfile
import base64

prediccion_valor = 0
prob_ingreso = 0

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Ingreso Universitario",
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title("🎓 Predictor de Ingreso Universitario")
st.markdown("**Sistema de predicción basado en Machine Learning**")
st.markdown("---")

# Funciones para cargar el modelo real entrenado y label encoders usados
@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo real entrenado
    """
    model = joblib.load('models/best_model.pkl')

    return model

@st.cache_resource
def cargar_label_encoder():
    """
    Carga label encoders usados
    """
    label_encoders = joblib.load('models/label_encoders.pkl')
    return label_encoders

# Función para preprocesar datos usando el modelo real
def preprocesar_datos(datos, label_encoders):
    """Convierte los datos del formulario usando los label encoders guardados"""
    
    # Crear DataFrame con el formato correcto
    data_df = pd.DataFrame({
        'COLEGIO_ANIO_EGRESO': [datos['año_egreso']],
        'ANIO_POSTULA': [datos['año_postulacion']],
        'CICLO_POSTULA': [1 if datos['ciclo'] == 'I Ciclo' else 2],
        'ANIO_NACIMIENTO': [datos['año_nacimiento']],
        'CALIF_FINAL': [datos['calificacion_final']],
        'COLEGIO': [datos['colegio']],
        'ESPECIALIDAD': [datos['especialidad']],
        'SEXO': [datos['sexo']],
        'MODALIDAD': [datos['modalidad']],
    })
    
    categorical_columns = ['COLEGIO', 'ESPECIALIDAD', 'SEXO', 'MODALIDAD']
    
    for col in categorical_columns:
        le = label_encoders[col]
        data_df[col + "_ENCODED"] = data_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        data_df = data_df.drop(col, axis=1)
    
    return data_df

# Función para validar datos
def validar_datos(datos):
    errores = []

    if datos['año_nacimiento'] > datetime.now().year - 15:
        errores.append("El año de nacimiento debe ser al menos 15 años atrás")

    if datos['año_egreso'] > datetime.now().year:
        errores.append("El año de egreso no puede ser futuro")

    if datos['año_egreso'] < datos['año_nacimiento'] + 15:
        errores.append("El año de egreso debe ser al menos 15 años después del nacimiento")

    if datos['año_postulacion'] < datos['año_egreso']:
        errores.append("El año de postulación no puede ser anterior al año de egreso")

    if datos['calificacion_final'] < 0 or datos['calificacion_final'] > 20:
        errores.append("La calificación debe estar entre 0 y 20")

    return errores

# Cargar modelo real
model = cargar_modelo()

# Cargar label encoders
label_encoders = cargar_label_encoder()

def generar_pdf_formulario(datos, prediccion):
    """
    Genera un PDF con los datos del formulario y la predicción
    """
    
    # Crear PDF
    doc, story = create_pdf("reporte_prediccion")
    
    # Título principal
    add_title(story, "Predictor de Ingreso Universitario")
    add_spacer(story, 1, 12)
    
    # Información del estudiante
    add_subtitle(story, "Información del Estudiante")
    datos_tabla = [
        ["Campo", "Valor"],
        ["Año de Nacimiento", str(datos['año_nacimiento'])],
        ["Sexo", str(datos['sexo'])],
        ["Colegio", str(datos['colegio'])],
        ["Año de Egreso", str(datos['año_egreso'])],
        ["Especialidad", str(datos['especialidad'])],
        ["Año de Postulación", str(datos['año_postulacion'])],
        ["Ciclo", str(datos['ciclo'])],
        ["Modalidad", str(datos['modalidad'])],
        ["Calificación Final", f"{datos['calificacion_final']:.1f}"]
    ]
    add_spacer(story, 1, 6)
    
    # Crear DataFrame para la tabla
    df_datos = pd.DataFrame(datos_tabla[1:], columns=datos_tabla[0])
    add_table(story, df_datos)
    add_spacer(story, 1, 6)
    
    # Resultado de predicción si existe
    if prediccion is not None:
        add_subtitle(story, "Resultado de la Predicción")
        
        prob_ingreso = prediccion * 100
        resultado = "INGRESO PROBABLE" if prediccion > 0.5 else "INGRESO POCO PROBABLE"
        
        add_paragraph(story, f"<b>Resultado:</b> {resultado}")
        add_paragraph(story, f"<b>Probabilidad de Ingreso:</b> {prob_ingreso:.1f}%")
        add_spacer(story, 1, 6)
    
    # Información del modelo
    add_subtitle(story, "Información del Modelo")
    add_paragraph(story, "• Modelo: Bosques Aleatorios")
    add_paragraph(story, "• R² Score: 86.31%")
    add_paragraph(story, "• Precisión: 91.97%")
    add_paragraph(story, "• Características: 9 variables")
    
    # Fecha de generación
    add_spacer(story, 1, 200)
    add_paragraph(story, f"<i>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>")
    
    # Construir PDF
    build_pdf(doc, story)
    
    return "reporte_prediccion.pdf"

def get_pdf_download_link(pdf_path, filename):
    """
    Genera un enlace de descarga para el PDF
    """
    with open(pdf_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Descargar PDF</a>'
        return href

# Crear el formulario
with st.form("formulario_prediccion"):
    st.subheader("📋 Información del Estudiante")

    col1, col2 = st.columns(2)

    with col1:
        año_nacimiento = st.number_input(
            "Año de Nacimiento",
            min_value=1950,
            max_value=datetime.now().year - 15,
            value=2000,
            step=1
        )

        sexo = st.selectbox(
            "Sexo",
            ["MASCULINO", "F"]
        )

        colegio = st.selectbox(
            "Nombre del Colegio",
            [
                "LA DIVINA PROVIDENCIA",
                "86019 LA LIBERTAD",
                "0113 DANIEL ALOMIAS ROBLES",
                "SEBASTIAN SALAZAR BONDY",
                "TRILCE LOS OLIVOS",
                "BARTOLOME HERRERA",
                "FE Y ALEGRIA 59",
                "CIENCIAS",
                "SAN CARLOS",
                "TRILCE SAN JUAN",
                "OTRO"
            ]
        )

        año_egreso = st.number_input(
            "Año de Egreso del Colegio",
            min_value=1970,
            max_value=datetime.now().year,
            value=2018,
            step=1
        )

    with col2:
        especialidad = st.selectbox(
            "Especialidad Universitaria",
            [
                "INGENIERÍA DE SISTEMAS",
                "INGENIERÍA DE TELECOMUNICACIONES", 
                "INGENIERÍA MECÁNICA",
                "INGENIERÍA ELECTRÓNICA",
                "ARQUITECTURA",
                "MEDICINA",
                "DERECHO",
                "ADMINISTRACIÓN",
                "CONTABILIDAD",
                "OTRO"
            ]
        )

        año_postulacion = st.number_input(
            "Año de Postulación",
            min_value=2000,
            max_value=datetime.now().year + 2,
            value=datetime.now().year,
            step=1
        )

        ciclo = st.selectbox(
            "Ciclo del Año",
            ["I Ciclo", "II Ciclo"]
        )

        modalidad = st.selectbox(
            "Modalidad de Postulación",
            [
                "ORDINARIO",
                "EXTRAORDINARIO1 - DEPORTISTAS CALIFICADOS DE ALTO NIVEL( Iniciar estudios)",
                "EXTRAORDINARIO2 – INGRESO DIRECTO CEPRE",
                "EXTRAORDINARIO1 - CONVENIO ANDRES BELLO (iniciar estudios)",
                "EXTRAORDINARIO INGRESO DIRECTO CEPRE-UNI",
                "EXTRAORDINARIO - DOS PRIMEROS ALUMNOS",
                "TALENTO BECA 18",
                "INGRESO ESCOLAR NACIONAL"
            ]
        )

    st.subheader("📊 Rendimiento Académico")
    calificacion_final = st.number_input(
        "Calificación Final",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        format="%.1f",
        help="Calificación obtenida en el proceso de admisión"
    )

    # Botones del formulario
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predecir = st.form_submit_button("🔮 Predecir Ingreso", type="primary")

    with col_btn2:
        generar_pdf = st.form_submit_button("📄 Generar PDF", type="secondary")

    # Procesamiento del formulario
    if predecir or generar_pdf:
        # Recopilar datos
        datos = {
            'año_nacimiento': año_nacimiento,
            'sexo': sexo,
            'colegio': colegio,
            'año_egreso': año_egreso,
            'especialidad': especialidad,
            'año_postulacion': año_postulacion,
            'ciclo': ciclo,
            'modalidad': modalidad,
            'calificacion_final': calificacion_final
        }

        # Validar datos
        errores = validar_datos(datos)

        if errores:
            st.error("❌ Se encontraron los siguientes errores:")
            for error in errores:
                st.error(f"• {error}")
        else:
            # Campos obligatorios
            if not colegio.strip() or not especialidad.strip():
                st.error("❌ Por favor, complete todos los campos obligatorios")
            else:
                st.success("✅ Datos procesados exitosamente!")

                # Realizar predicción si se solicitó
                if predecir:
                    st.markdown("---")
                    st.subheader("🔮 Resultado de la Predicción")

                    try:
                        # Preprocesar datos para el modelo
                        caracteristicas = preprocesar_datos(datos, label_encoders)

                        # Realizar predicción con el modelo real
                        prediccion_valor = model.predict(caracteristicas)[0]

                        prob_ingreso = prediccion_valor * 100

                        # Mostrar resultado
                        col_pred1, col_pred2 = st.columns(2)

                        with col_pred1:
                            if prediccion_valor > 0.5:
                                st.success("🎉 **INGRESO PROBABLE**")
                                st.balloons()
                            else:
                                st.error("❌ **INGRESO POCO PROBABLE**")

                        with col_pred2:
                            st.metric(
                                "Probabilidad de Ingreso",
                                f"{prob_ingreso:.1f}%",
                                delta=f"{prob_ingreso-50:.1f}%" if prob_ingreso > 50 else None
                            )
                        
                        # Mostrar información del modelo
                        st.info(f"🤖 Modelo utilizado: {type(model).__name__} (R² = 86.31%)")
                        
                    except Exception as e:
                        st.error(f"❌ Error en la predicción: {e}")
                        st.info("💡 Asegúrate de que los valores ingresados sean válidos para el modelo entrenado.")

                    # Mostrar factores influyentes
                    st.subheader("📈 Análisis de Factores")

                    # Crear análisis básico
                    factores = []
                    if calificacion_final >= 15:
                        factores.append("✅ Excelente calificación (≥15)")
                    elif calificacion_final >= 12:
                        factores.append("⚠️ Calificación promedio (12-14)")
                    else:
                        factores.append("❌ Calificación baja (<12)")

                    if modalidad in ["Primeros Puestos", "Centro Pre-Universitario"]:
                        factores.append("✅ Modalidad favorable")

                    edad = datetime.now().year - año_nacimiento
                    if 17 <= edad <= 22:
                        factores.append("✅ Edad típica para postulación")

                    años_espera = año_postulacion - año_egreso
                    if años_espera <= 2:
                        factores.append("✅ Postulación temprana")
                    elif años_espera > 5:
                        factores.append("⚠️ Varios años desde egreso")

                    for factor in factores:
                        st.write(factor)

                # Generar PDF si se solicitó
                if generar_pdf:
                    st.markdown("---")
                    st.subheader("📄 Generación de PDF")
                    
                    try:
                        # Usar la predicción ya calculada
                        caracteristicas = preprocesar_datos(datos, label_encoders)
                        prediccion_valor = model.predict(caracteristicas)[0]
                        
                        # Generar PDF
                        pdf_path = generar_pdf_formulario(datos, prediccion_valor)
                        
                        # Crear nombre de archivo
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"prediccion_ingreso_{timestamp}.pdf"
                        
                        # Mostrar enlace de descarga
                        st.success("✅ PDF generado exitosamente!")
                        st.markdown(get_pdf_download_link(pdf_path, filename), unsafe_allow_html=True)
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(f"❌ Error al generar PDF: {e}")

                # Mostrar resumen de datos
                st.markdown("---")
                st.subheader("📋 Resumen de Datos")

                df_resumen = pd.DataFrame([
                    ["Año de Nacimiento", str(año_nacimiento)],
                    ["Sexo", str(sexo)],
                    ["Colegio", str(colegio)],
                    ["Año de Egreso", str(año_egreso)],
                    ["Especialidad", str(especialidad)],
                    ["Año de Postulación", str(año_postulacion)],
                    ["Ciclo", str(ciclo)],
                    ["Modalidad", str(modalidad)],
                    ["Calificación Final", f"{calificacion_final:.1f}"]
                ], columns=["Campo", "Valor"])

                # Ensure all values are strings to avoid PyArrow conversion issues
                df_resumen = df_resumen.astype(str)
                
                try:
                    # Use table instead of dataframe to avoid PyArrow issues
                    st.table(df_resumen)
                except Exception as e:
                    # Fallback display if table fails
                    st.write("**Resumen de Datos:**")
                    for _, row in df_resumen.iterrows():
                        st.write(f"• **{row['Campo']}:** {row['Valor']}")

# Sidebar con información
st.sidebar.header("ℹ️ Información del Modelo")
st.sidebar.markdown(f"""
**Modelo:** {type(model).__name__}\n
**R² Score:** 86.31%\n
**Precisión:** 91.97%\n
**Características:** 9 variables\n

**Mejor modelo entrenado con datos reales**
""")

# Botón para descargar el reporte de entrenamiento
with open("reporte.pdf", "rb") as f:
    st.sidebar.download_button(
        label="📄 Descargar Reporte de Entrenamiento",
        data=f,
        file_name="reporte.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")