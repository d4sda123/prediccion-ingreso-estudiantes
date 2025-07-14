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

# Language dictionary
LANG_DICT = {
    'es': {
        'main_title': "🎓 Predictor de Ingreso Universitario",
        'subtitle': "**Sistema de predicción basado en Machine Learning**",
        'student_info': "📋 Información del Estudiante",
        'year_of_birth': "Año de Nacimiento",
        'sex': "Sexo",
        'sexs': ['Masculino', 'Femenino'],
        'school_name': "Nombre del Colegio",
        'year_of_graduation': "Año de Egreso del Colegio",
        'university_specialty': "Especialidad Universitaria",
        'year_of_application': "Año de Postulación",
        'cycle': "Ciclo del Año",
        'cycles': ['I Ciclo', 'II Ciclo'],
        'application_mode': "Modalidad de Postulación",
        'academic_performance': "📊 Rendimiento Académico",
        'final_grade': "Calificación Final",
        'final_grade_help': "Calificación obtenida en el proceso de admisión",
        'predict_button': "🔮 Predecir Ingreso",
        'pdf_button': "📄 Generar PDF",
        'error_found': "❌ Se encontraron los siguientes errores:",
        'complete_fields': "❌ Por favor, complete todos los campos obligatorios",
        'success_data': "✅ Datos procesados exitosamente!",
        'prediction_result': "🔮 Resultado de la Predicción",
        'likely_admission': "🎉 **INGRESO PROBABLE**",
        'unlikely_admission': "❌ **INGRESO POCO PROBABLE**",
        'admission_probability': "Probabilidad de Ingreso",
        'model_info': "🤖 Modelo utilizado: {model} (R² = 86.31%)",
        'prediction_error': "❌ Error en la predicción: {e}",
        'prediction_hint': "💡 Asegúrate de que los valores ingresados sean válidos para el modelo entrenado.",
        'factor_analysis': "📈 Análisis de Factores",
        'excellent_grade': "✅ Excelente calificación (≥15)",
        'average_grade': "⚠️ Calificación promedio (12-14)",
        'low_grade': "❌ Calificación baja (<12)",
        'favorable_mode': "✅ Modalidad favorable",
        'typical_age': "✅ Edad típica para postulación",
        'early_application': "✅ Postulación temprana",
        'late_application': "⚠️ Varios años desde egreso",
        'pdf_generation': "📄 Generación de PDF",
        'pdf_success': "✅ PDF generado exitosamente!",
        'pdf_error': "❌ Error al generar PDF: {e}",
        'data_summary': "📋 Resumen de Datos",
        'sidebar_info': "ℹ️ Información del Modelo",
        'sidebar_model': "**Modelo:** {model}\n\n**R² Score:** 86.31%\n**Precisión:** 91.97%\n**Características:** 9 variables\n\n**Mejor modelo entrenado con datos reales**",
        'sidebar_report': "📄 Descargar Reporte de Entrenamiento",
        'footer': "---",
        'language': "Idioma",
        'spanish': "Español",
        'english': "Inglés"
    },
    'en': {
        'main_title': "🎓 University Admission Predictor",
        'subtitle': "**Prediction system based on Machine Learning**",
        'student_info': "📋 Student Information",
        'year_of_birth': "Year of Birth",
        'sex': "Sex",
        'sexs': ['Masculine', 'Femenine'],
        'school_name': "School Name",
        'year_of_graduation': "Year of Graduation",
        'university_specialty': "University Specialty",
        'year_of_application': "Year of Application",
        'cycle': "Year Cycle",
        'cycles': ['I Cycle', 'II Cycle'],
        'application_mode': "Application Mode",
        'academic_performance': "📊 Academic Performance",
        'final_grade': "Final Grade",
        'final_grade_help': "Grade obtained in the admission process",
        'predict_button': "🔮 Predict Admission",
        'pdf_button': "📄 Generate PDF",
        'error_found': "❌ The following errors were found:",
        'complete_fields': "❌ Please complete all required fields",
        'success_data': "✅ Data processed successfully!",
        'prediction_result': "🔮 Prediction Result",
        'likely_admission': "🎉 **LIKELY ADMISSION**",
        'unlikely_admission': "❌ **UNLIKELY ADMISSION**",
        'admission_probability': "Admission Probability",
        'model_info': "🤖 Model used: {model} (R² = 86.31%)",
        'prediction_error': "❌ Prediction error: {e}",
        'prediction_hint': "💡 Make sure the entered values are valid for the trained model.",
        'factor_analysis': "📈 Factor Analysis",
        'excellent_grade': "✅ Excellent grade (≥15)",
        'average_grade': "⚠️ Average grade (12-14)",
        'low_grade': "❌ Low grade (<12)",
        'favorable_mode': "✅ Favorable mode",
        'typical_age': "✅ Typical age for application",
        'early_application': "✅ Early application",
        'late_application': "⚠️ Several years since graduation",
        'pdf_generation': "📄 PDF Generation",
        'pdf_success': "✅ PDF generated successfully!",
        'pdf_error': "❌ Error generating PDF: {e}",
        'data_summary': "📋 Data Summary",
        'sidebar_info': "ℹ️ Model Information",
        'sidebar_model': "**Model:** {model}\n\n**R² Score:** 86.31%\n**Accuracy:** 91.97%\n**Features:** 9 variables\n\n**Best model trained with real data**",
        'sidebar_report': "📄 Download Training Report",
        'footer': "---",
        'language': "Language",
        'spanish': "Spanish",
        'english': "English"
    }
}

# Language selection (persist in session state)
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'es'
lang = st.sidebar.selectbox(
    LANG_DICT[st.session_state['lang']]['language'],
    options=['es', 'en'],
    format_func=lambda x: LANG_DICT[x]['spanish'] if x == 'es' else LANG_DICT[x]['english'],
    key='lang'
)
lang_dict = LANG_DICT[st.session_state['lang']]

# Configuración de la página
st.set_page_config(
    page_title=lang_dict['main_title'],
    page_icon="🎓",
    layout="wide"
)

# Título principal
st.title(lang_dict['main_title'])
st.markdown(lang_dict['subtitle'])
st.markdown(lang_dict['footer'])

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
    st.subheader(lang_dict['student_info'])

    col1, col2 = st.columns(2)

    with col1:
        año_nacimiento = st.number_input(
            lang_dict['year_of_birth'],
            min_value=1950,
            max_value=datetime.now().year - 15,
            value=2000,
            step=1
        )

        sexo = st.selectbox(
            lang_dict['sex'],
            lang_dict['sexs']
        )

        colegio = st.selectbox(
            lang_dict['school_name'],
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
            lang_dict['year_of_graduation'],
            min_value=1970,
            max_value=datetime.now().year,
            value=2018,
            step=1
        )

    with col2:
        especialidad = st.selectbox(
            lang_dict['university_specialty'],
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
            lang_dict['year_of_application'],
            min_value=2000,
            max_value=datetime.now().year + 2,
            value=datetime.now().year,
            step=1
        )

        ciclo = st.selectbox(
            lang_dict['cycle'],
            lang_dict['cycles']
        )

        modalidad = st.selectbox(
            lang_dict['application_mode'],
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

    st.subheader(lang_dict['academic_performance'])
    calificacion_final = st.number_input(
        lang_dict['final_grade'],
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        format="%.1f",
        help=lang_dict['final_grade_help']
    )

    # Botones del formulario
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predecir = st.form_submit_button(lang_dict['predict_button'], type="primary")

    with col_btn2:
        generar_pdf = st.form_submit_button(lang_dict['pdf_button'], type="secondary")

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
            st.error(lang_dict['error_found'])
            for error in errores:
                st.error(f"• {error}")
        else:
            # Campos obligatorios
            if not colegio.strip() or not especialidad.strip():
                st.error(lang_dict['complete_fields'])
            else:
                st.success(lang_dict['success_data'])

                # Realizar predicción si se solicitó
                if predecir:
                    st.markdown(lang_dict['footer'])
                    st.subheader(lang_dict['prediction_result'])

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
                                st.success(lang_dict['likely_admission'])
                                st.balloons()
                            else:
                                st.error(lang_dict['unlikely_admission'])

                        with col_pred2:
                            st.metric(
                                lang_dict['admission_probability'],
                                f"{prob_ingreso:.1f}%",
                                delta=f"{prob_ingreso-50:.1f}%" if prob_ingreso > 50 else None
                            )
                        
                        # Mostrar información del modelo
                        st.info(lang_dict['model_info'].format(model=type(model).__name__))
                        
                    except Exception as e:
                        st.error(lang_dict['prediction_error'].format(e=e))
                        st.info(lang_dict['prediction_hint'])

                    # Mostrar factores influyentes
                    st.subheader(lang_dict['factor_analysis'])

                    # Crear análisis básico
                    factores = []
                    if calificacion_final >= 15:
                        factores.append(lang_dict['excellent_grade'])
                    elif calificacion_final >= 12:
                        factores.append(lang_dict['average_grade'])
                    else:
                        factores.append(lang_dict['low_grade'])

                    if modalidad in ["Primeros Puestos", "Centro Pre-Universitario"]:
                        factores.append(lang_dict['favorable_mode'])

                    edad = datetime.now().year - año_nacimiento
                    if 17 <= edad <= 22:
                        factores.append(lang_dict['typical_age'])

                    años_espera = año_postulacion - año_egreso
                    if años_espera <= 2:
                        factores.append(lang_dict['early_application'])
                    elif años_espera > 5:
                        factores.append(lang_dict['late_application'])

                    for factor in factores:
                        st.write(factor)

                # Generar PDF si se solicitó
                if generar_pdf:
                    st.markdown(lang_dict['footer'])
                    st.subheader(lang_dict['pdf_generation'])
                    
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
                        st.success(lang_dict['pdf_success'])
                        st.markdown(get_pdf_download_link(pdf_path, filename), unsafe_allow_html=True)
                        
                        # Limpiar archivo temporal
                        try:
                            os.unlink(pdf_path)
                        except:
                            pass
                            
                    except Exception as e:
                        st.error(lang_dict['pdf_error'].format(e=e))

                # Mostrar resumen de datos
                st.markdown(lang_dict['footer'])
                st.subheader(lang_dict['data_summary'])

                df_resumen = pd.DataFrame.from_records(
                    [
                        [lang_dict['year_of_birth'], str(año_nacimiento)],
                        [lang_dict['sex'], str(sexo)],
                        [lang_dict['school_name'], str(colegio)],
                        [lang_dict['year_of_graduation'], str(año_egreso)],
                        [lang_dict['university_specialty'], str(especialidad)],
                        [lang_dict['year_of_application'], str(año_postulacion)],
                        [lang_dict['cycle'], str(ciclo)],
                        [lang_dict['application_mode'], str(modalidad)],
                        [lang_dict['final_grade'], f"{calificacion_final:.1f}"]
                    ],
                    columns=("Campo", "Valor")
                )
                
                try:
                    # Use table instead of dataframe to avoid PyArrow issues
                    st.table(df_resumen)
                except Exception as e:
                    # Fallback display if table fails
                    st.write("**Resumen de Datos:**")
                    for _, row in df_resumen.iterrows():
                        st.write(f"• **{row['Campo']}:** {row['Valor']}")

# Sidebar con información
st.sidebar.header(lang_dict['sidebar_info'])
st.sidebar.markdown(lang_dict['sidebar_model'].format(model=type(model).__name__))

# Botón para descargar el reporte de entrenamiento
with open("reporte.pdf", "rb") as f:
    st.sidebar.download_button(
        label=lang_dict['sidebar_report'],
        data=f,
        file_name="reporte.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown(lang_dict['footer'])