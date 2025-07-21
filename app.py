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
        'english': "Inglés",
        'pdf_download': "Descargar PDF"
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
        'english': "English",
        'pdf_download': "Download PDF"
    }
}

# Import additional languages
from languages import additional_languages, get_all_supported_languages, get_language_name
from option_translations import (get_translated_schools, get_translated_specialties, 
                                get_translated_application_modes, get_original_value,
                                SCHOOL_TRANSLATIONS, SPECIALTY_TRANSLATIONS, APPLICATION_MODE_TRANSLATIONS)

# Merge additional languages into LANG_DICT
LANG_DICT.update(additional_languages)

# Add missing keys to Spanish and English dictionaries
LANG_DICT['es']['value'] = 'Valor'
LANG_DICT['en']['value'] = 'Value'
LANG_DICT['es']['french'] = 'Francés'
LANG_DICT['es']['german'] = 'Alemán'
LANG_DICT['es']['italian'] = 'Italiano'
LANG_DICT['es']['portuguese'] = 'Portugués'
LANG_DICT['es']['russian'] = 'Ruso'
LANG_DICT['es']['chinese'] = 'Chino'
LANG_DICT['es']['japanese'] = 'Japonés'
LANG_DICT['en']['french'] = 'French'
LANG_DICT['en']['german'] = 'German'
LANG_DICT['en']['italian'] = 'Italian'
LANG_DICT['en']['portuguese'] = 'Portuguese'
LANG_DICT['en']['russian'] = 'Russian'
LANG_DICT['en']['chinese'] = 'Chinese'
LANG_DICT['en']['japanese'] = 'Japanese'

# Language selection (persist in session state)
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'es'

# Get current language for selector label
current_lang_dict = LANG_DICT.get(st.session_state['lang'], LANG_DICT['es'])

# Define callback for language change
def on_language_change():
    st.session_state['lang'] = st.session_state['lang_selector']

lang = st.sidebar.selectbox(
    current_lang_dict['language'],
    options=get_all_supported_languages(),
    format_func=lambda x: get_language_name(x),
    index=get_all_supported_languages().index(st.session_state['lang']),
    key='lang_selector',
    on_change=on_language_change
)

# Update session state
st.session_state['lang'] = lang
lang_dict = LANG_DICT[lang]

# Configuración de la página
st.set_page_config(
    page_title=lang_dict['main_title'],
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    /* Estilo para el contenedor principal */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Estilo para tarjetas de sección */
    .section-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Estilo para tarjetas de información */
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
        backdrop-filter: blur(5px);
    }
    
    /* Mejora de botones */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        height: 3rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Estilo para el título principal */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Estilo para subtítulos */
    .section-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Indicador de progreso */
    .progress-container {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e1e5e9;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Alertas personalizadas */
    .custom-alert {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-success {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    /* Animaciones */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .section-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Título principal mejorado
st.markdown(f'<h1 class="main-title">{lang_dict["main_title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">{lang_dict["subtitle"]}</div>', unsafe_allow_html=True)

# Cargar modelo entrenado directamente (sin archivo pkl por ahora)
@st.cache_resource
def cargar_modelo():
    """
    Como no tenemos el modelo entrenado, crearemos uno simulado para demostración
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    # Cargar datos del student-por.csv
    df = pd.read_csv('student-por.csv')
    
    # Preparar datos
    categorical_cols = ['school', 'sex', 'address', 'Mjob', 'reason', 'higher', 'internet']
    
    # Crear encoders para variables categóricas
    encoded_df = df.copy()
    
    for col in categorical_cols:
        encoded_df[col] = pd.Categorical(encoded_df[col]).codes
    
    # Features (19 variables)
    feature_columns = ['school', 'sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 
                      'reason', 'traveltime', 'studytime', 'failures', 'higher', 
                      'internet', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']
    
    X = encoded_df[feature_columns]
    y = encoded_df['G3']  # Calificación final como target
    
    # Entrenar modelo simple
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, encoded_df[categorical_cols].apply(lambda x: pd.Categorical(x)).to_dict()

# Función para preprocesar datos del formulario
def preprocesar_datos_student(datos):
    """Convierte los datos del formulario al formato esperado por el modelo"""
    
    # Mapear valores categóricos a códigos numéricos
    school_map = {'GP': 0, 'MS': 1}
    sex_map = {'F': 0, 'M': 1}
    address_map = {'U': 0, 'R': 1}
    mjob_map = {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4}
    reason_map = {'course': 0, 'home': 1, 'other': 2, 'reputation': 3}
    binary_map = {'no': 0, 'yes': 1}
    
    # Crear array con las 19 características
    features = np.array([
        school_map[datos['school']],
        sex_map[datos['sex']],
        datos['age'],
        address_map[datos['address']],
        datos['Medu'],
        datos['Fedu'],
        mjob_map[datos['Mjob']],
        reason_map[datos['reason']],
        datos['traveltime'],
        datos['studytime'],
        datos['failures'],
        binary_map[datos['higher']],
        binary_map[datos['internet']],
        datos['Dalc'],
        datos['Walc'],
        datos['health'],
        datos['absences'],
        datos['G1'],
        datos['G2']
    ]).reshape(1, -1)
    
    return features

# Función para validar datos del student-por
def validar_datos_student(datos):
    errores = []

    if datos['age'] < 15 or datos['age'] > 22:
        errores.append("La edad debe estar entre 15 y 22 años")

    if datos['Medu'] < 0 or datos['Medu'] > 4:
        errores.append("El nivel de educación de la madre debe estar entre 0 y 4")
        
    if datos['Fedu'] < 0 or datos['Fedu'] > 4:
        errores.append("El nivel de educación del padre debe estar entre 0 y 4")

    if datos['failures'] < 0 or datos['failures'] > 4:
        errores.append("El número de fallas debe estar entre 0 y 4")

    if datos['G1'] < 0 or datos['G1'] > 20:
        errores.append("La calificación G1 debe estar entre 0 y 20")
        
    if datos['G2'] < 0 or datos['G2'] > 20:
        errores.append("La calificación G2 debe estar entre 0 y 20")
        
    if datos['absences'] < 0 or datos['absences'] > 93:
        errores.append("El número de ausencias debe estar entre 0 y 93")

    return errores

# Cargar modelo
model, categorical_encoders = cargar_modelo()

def generar_pdf_formulario(datos, prediccion):
    """
    Genera un PDF con los datos del formulario y la predicción
    """
    
    lang_dict = LANG_DICT[st.session_state['lang']]

    # Crear PDF
    doc, story = create_pdf("reporte_prediccion")
    
    # Título principal
    add_title(story, lang_dict['main_title'])
    add_spacer(story, 1, 12)
    
    # Información del estudiante
    add_subtitle(story, lang_dict['student_info'])
    datos_tabla = [
        [lang_dict['data_summary'], lang_dict['value']],
        [lang_dict['year_of_birth'], str(datos['año_nacimiento'])],
        [lang_dict['sex'], str(datos['sexo'])],
        [lang_dict['school_name'], str(datos['colegio'])],
        [lang_dict['year_of_graduation'], str(datos['año_egreso'])],
        [lang_dict['university_specialty'], str(datos['especialidad'])],
        [lang_dict['year_of_application'], str(datos['año_postulacion'])],
        [lang_dict['cycle'], str(datos['ciclo'])],
        [lang_dict['application_mode'], str(datos['modalidad'])],
        [lang_dict['final_grade'], f"{datos['calificacion_final']:.1f}"]
    ]
    add_spacer(story, 1, 6)
    
    # Crear DataFrame para la tabla
    df_datos = pd.DataFrame(datos_tabla[1:], columns=datos_tabla[0])
    add_table(story, df_datos)
    add_spacer(story, 1, 6)
    
    # Resultado de predicción si existe
    if prediccion is not None:
        add_subtitle(story, lang_dict['prediction_result'])
        
        prob_ingreso = prediccion * 100
        resultado = lang_dict['likely_admission'] if prediccion > 0.5 else lang_dict['unlikely_admission']
        
        add_paragraph(story, f"<b>{lang_dict['prediction_result']}:</b> {resultado}")
        add_paragraph(story, f"<b>{lang_dict['admission_probability']}:</b> {prob_ingreso:.1f}%")
        add_spacer(story, 1, 6)
    
    # Información del modelo
    add_subtitle(story, lang_dict['sidebar_info'])
    add_paragraph(story, lang_dict['sidebar_model'].format(model='Random Forest'))
    add_spacer(story, 1, 6)
    
    # Fecha de generación
    add_spacer(story, 1, 200)
    generation_text = {
        'es': f"<i>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'en': f"<i>Generated on: {datetime.now().strftime('%m/%d/%Y %H:%M:%S')}</i>",
        'fr': f"<i>Généré le: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'de': f"<i>Generiert am: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'it': f"<i>Generato il: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'pt': f"<i>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'ru': f"<i>Создано: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>",
        'zh': f"<i>生成于: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}</i>",
        'ja': f"<i>生成日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}</i>"
    }
    current_lang = st.session_state['lang']
    add_paragraph(story, generation_text.get(current_lang, generation_text['es']))
    
    # Construir PDF
    build_pdf(doc, story)
    
    return "reporte_prediccion.pdf"

def get_pdf_download_link(pdf_path, filename):
    """
    Genera un enlace de descarga para el PDF
    """
    lang_dict = LANG_DICT[st.session_state['lang']]
    with open(pdf_path, "rb") as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{lang_dict["pdf_download"]}</a>'
        return href

# Instrucciones paso a paso
st.markdown(
    f"""
    <div class="info-card">
        <h4 style="margin-top: 0; color: #2c3e50;">📝 Instrucciones</h4>
        <p style="margin-bottom: 0;">Complete el formulario con la información solicitada para obtener una predicción precisa sobre el rendimiento académico.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Crear el formulario con las 19 características del student-por.csv
with st.form("formulario_prediccion"):
    # Sección 1: Información Básica del Estudiante
    st.markdown(
        f"""
        <div class="section-card fade-in">
            <h3 style="margin-top: 0; color: white;">📋 Información Básica del Estudiante</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 1. School
        school = st.selectbox(
            "🏫 Escuela",
            ["GP", "MS"],
            format_func=lambda x: "Gabriel Pereira" if x == "GP" else "Mousinho da Silveira",
            help="Escuela del estudiante"
        )
        
        # 2. Sex
        sex = st.selectbox(
            "👤 Sexo",
            ["F", "M"],
            format_func=lambda x: "Femenino" if x == "F" else "Masculino"
        )
        
        # 3. Age
        age = st.number_input(
            "🎂 Edad",
            min_value=15,
            max_value=22,
            value=16,
            help="Edad actual del estudiante"
        )
        
    with col2:
        # 4. Address
        address = st.selectbox(
            "🏠 Tipo de Dirección",
            ["U", "R"],
            format_func=lambda x: "Urbana" if x == "U" else "Rural"
        )
        
        # 5. Mother's Education
        Medu = st.selectbox(
            "👩‍🎓 Educación de la Madre",
            [0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Sin educación",
                1: "Educación primaria (4to grado)",
                2: "5to a 9no grado",
                3: "Educación secundaria",
                4: "Educación superior"
            }[x]
        )
        
        # 6. Father's Education
        Fedu = st.selectbox(
            "👨‍🎓 Educación del Padre",
            [0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: "Sin educación",
                1: "Educación primaria (4to grado)",
                2: "5to a 9no grado",
                3: "Educación secundaria",
                4: "Educación superior"
            }[x]
        )
        
    with col3:
        # 7. Mother's Job
        Mjob = st.selectbox(
            "👩‍💼 Trabajo de la Madre",
            ["teacher", "health", "services", "at_home", "other"],
            format_func=lambda x: {
                "teacher": "Profesora",
                "health": "Salud",
                "services": "Servicios",
                "at_home": "En casa",
                "other": "Otro"
            }[x]
        )
        
        # 8. Reason for choosing school
        reason = st.selectbox(
            "🤔 Razón para elegir la escuela",
            ["home", "reputation", "course", "other"],
            format_func=lambda x: {
                "home": "Cerca de casa",
                "reputation": "Reputación de la escuela",
                "course": "Preferencia del curso",
                "other": "Otro"
            }[x]
        )
        
        # 9. Travel time
        traveltime = st.selectbox(
            "🚌 Tiempo de viaje a la escuela",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "<15 min",
                2: "15-30 min",
                3: "30 min - 1 hora",
                4: ">1 hora"
            }[x]
        )
    
    # Separador
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Sección 2: Información Académica y Social
    st.markdown(
        f"""
        <div class="section-card fade-in">
            <h3 style="margin-top: 0; color: white;">📚 Información Académica y Social</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # 10. Study time
        studytime = st.selectbox(
            "⏱️ Tiempo de estudio semanal",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "<2 horas",
                2: "2-5 horas",
                3: "5-10 horas",
                4: ">10 horas"
            }[x]
        )
        
        # 11. Past class failures
        failures = st.number_input(
            "❌ Fallas académicas previas",
            min_value=0,
            max_value=4,
            value=0,
            help="Número de fallas en clases anteriores"
        )
        
        # 12. Higher education support
        higher = st.selectbox(
            "🎓 ¿Quiere seguir educación superior?",
            ["yes", "no"],
            format_func=lambda x: "Sí" if x == "yes" else "No"
        )
        
    with col5:
        # 13. Internet access
        internet = st.selectbox(
            "🌐 Acceso a Internet en casa",
            ["yes", "no"],
            format_func=lambda x: "Sí" if x == "yes" else "No"
        )
        
        # 14. Workday alcohol consumption
        Dalc = st.selectbox(
            "🍺 Consumo de alcohol entre semana",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Muy bajo",
                2: "Bajo",
                3: "Medio",
                4: "Alto",
                5: "Muy alto"
            }[x]
        )
        
        # 15. Weekend alcohol consumption
        Walc = st.selectbox(
            "🍻 Consumo de alcohol fin de semana",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Muy bajo",
                2: "Bajo",
                3: "Medio",
                4: "Alto",
                5: "Muy alto"
            }[x]
        )
        
    with col6:
        # 16. Health status
        health = st.selectbox(
            "🏥 Estado de salud actual",
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Muy malo",
                2: "Malo",
                3: "Regular",
                4: "Bueno",
                5: "Muy bueno"
            }[x],
            index=4  # Default to "Bueno"
        )
        
        # 17. Absences
        absences = st.number_input(
            "🚫 Número de ausencias escolares",
            min_value=0,
            max_value=93,
            value=0,
            help="Número total de ausencias"
        )
    
    # Separador
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Sección 3: Calificaciones
    st.markdown(
        f"""
        <div class="section-card fade-in">
            <h3 style="margin-top: 0; color: white;">📊 Calificaciones</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col7, col8 = st.columns(2)
    
    with col7:
        # 18. G1 - first period grade
        G1 = st.number_input(
            "📈 Calificación 1er Período (G1)",
            min_value=0,
            max_value=20,
            value=0,
            help="Calificación del primer período (0-20)"
        )
        
    with col8:
        # 19. G2 - second period grade
        G2 = st.number_input(
            "📈 Calificación 2do Período (G2)",
            min_value=0,
            max_value=20,
            value=0,
            help="Calificación del segundo período (0-20)"
        )
    
    # Indicadores visuales de las calificaciones
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        if G1 >= 15:
            st.success("✅ G1: Excelente calificación")
        elif G1 >= 10:
            st.warning("⚠️ G1: Calificación promedio")
        elif G1 > 0:
            st.error("❌ G1: Calificación baja")
            
    with col_ind2:
        if G2 >= 15:
            st.success("✅ G2: Excelente calificación")
        elif G2 >= 10:
            st.warning("⚠️ G2: Calificación promedio")
        elif G2 > 0:
            st.error("❌ G2: Calificación baja")
    
    # Separador
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # Botones del formulario con mejor diseño
    st.markdown(
        """
        <div style="text-align: center; margin: 2rem 0;">
            <h4 style="color: #2c3e50; margin-bottom: 1.5rem;">✨ ¿Listo para conocer el resultado?</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_btn1, col_btn2 = st.columns(2, gap="large")

    with col_btn1:
        predecir = st.form_submit_button(
            "🔮 Predecir Calificación Final", 
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        generar_pdf = st.form_submit_button(
            "📄 Generar Reporte PDF", 
            type="secondary",
            use_container_width=True
        )

    # Procesamiento del formulario
    if predecir or generar_pdf:
        # Recopilar datos del formulario student-por
        datos_student = {
            'school': school,
            'sex': sex,
            'age': age,
            'address': address,
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': Mjob,
            'reason': reason,
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'higher': higher,
            'internet': internet,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'absences': absences,
            'G1': G1,
            'G2': G2
        }

        # Validar datos
        errores = validar_datos_student(datos_student)

        if errores:
            st.error("❌ Se encontraron los siguientes errores:")
            for error in errores:
                st.error(f"• {error}")
        else:
            st.success("✅ Datos procesados exitosamente!")

            # Realizar predicción si se solicitó
            if predecir:
                st.markdown("---")
                st.subheader("🔮 Resultado de la Predicción")

                try:
                    # Preprocesar datos para el modelo
                    caracteristicas = preprocesar_datos_student(datos_student)

                    # Realizar predicción con el modelo
                    prediccion_valor = model.predict(caracteristicas)[0]

                    # Mostrar resultado
                    col_pred1, col_pred2 = st.columns(2)

                    with col_pred1:
                        if prediccion_valor >= 15:
                            st.success("🎉 **EXCELENTE PREDICCIÓN**")
                            st.balloons()
                        elif prediccion_valor >= 10:
                            st.warning("⚠️ **PREDICCIÓN PROMEDIO**")
                        else:
                            st.error("❌ **PREDICCIÓN BAJA**")

                    with col_pred2:
                        st.metric(
                            "Calificación Final Predicha (G3)",
                            f"{prediccion_valor:.1f}/20",
                            delta=f"{prediccion_valor-10:.1f}" if prediccion_valor != 10 else None
                        )
                    
                    # Mostrar información del modelo
                    st.info(f"🤖 Modelo utilizado: {type(model).__name__}")
                    
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {e}")
                    st.info("💡 Asegúrate de que los valores ingresados sean válidos.")

                # Mostrar factores influyentes
                st.subheader("📈 Análisis de Factores")

                # Crear análisis básico
                factores = []
                if G1 >= 15 or G2 >= 15:
                    factores.append("✅ Excelentes calificaciones previas")
                elif G1 >= 10 or G2 >= 10:
                    factores.append("⚠️ Calificaciones promedio previas")
                else:
                    factores.append("❌ Calificaciones bajas previas")

                if higher == "yes":
                    factores.append("✅ Interés en educación superior")

                if age >= 15 and age <= 18:
                    factores.append("✅ Edad típica para el nivel")

                if failures == 0:
                    factores.append("✅ Sin fallas académicas previas")
                elif failures > 2:
                    factores.append("❌ Múltiples fallas académicas")

                if studytime >= 3:
                    factores.append("✅ Buen tiempo de estudio")
                    
                if internet == "yes":
                    factores.append("✅ Acceso a Internet")

                for factor in factores:
                    st.write(factor)

            # Generar PDF si se solicitó
            if generar_pdf:
                st.markdown("---")
                st.subheader("📄 Generación de PDF")
                
                try:
                    # Usar la predicción ya calculada
                    caracteristicas = preprocesar_datos_student(datos_student)
                    prediccion_valor = model.predict(caracteristicas)[0]
                    
                    # Generar PDF
                    pdf_path = generar_pdf_student(datos_student, prediccion_valor)
                    
                    # Crear nombre de archivo
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"prediccion_academica_{timestamp}.pdf"
                    
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

            df_resumen = pd.DataFrame.from_records(
                [
                    ["Escuela", "Gabriel Pereira" if school == "GP" else "Mousinho da Silveira"],
                    ["Sexo", "Femenino" if sex == "F" else "Masculino"],
                    ["Edad", str(age)],
                    ["Tipo de Dirección", "Urbana" if address == "U" else "Rural"],
                    ["Educación Madre", str(Medu)],
                    ["Educación Padre", str(Fedu)],
                    ["Trabajo Madre", Mjob],
                    ["Razón Escuela", reason],
                    ["Tiempo Viaje", str(traveltime)],
                    ["Tiempo Estudio", str(studytime)],
                    ["Fallas Previas", str(failures)],
                    ["Educación Superior", "Sí" if higher == "yes" else "No"],
                    ["Internet", "Sí" if internet == "yes" else "No"],
                    ["Consumo Alcohol Semanal", str(Dalc)],
                    ["Consumo Alcohol Fin de Semana", str(Walc)],
                    ["Estado de Salud", str(health)],
                    ["Ausencias", str(absences)],
                    ["Calificación G1", f"{G1:.1f}"],
                    ["Calificación G2", f"{G2:.1f}"]
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