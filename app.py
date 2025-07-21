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
        'pdf_generation': "📄 Generación de PDF",
        'pdf_success': "✅ PDF generado exitosamente!",
        'pdf_error': "❌ Error al generar PDF: {e}",
        'data_summary': "📋 Resumen de Datos",
        'sidebar_info': "ℹ️ Información del Modelo",
        'sidebar_model': "**Modelo:** {model}\n\n**R² Score:** 86.31%\n**Precisión:** 91.97%\n**Características:** 19 variables\n\n**Mejor modelo entrenado con datos reales**",
        'sidebar_report': "📄 Descargar Reporte de Entrenamiento",
        'footer': "---",
        'language': "Idioma",
        'spanish': "Español",
        'english': "Inglés",
        'pdf_download': "Descargar PDF",
        'value': 'Valor',
        'french': 'Francés',
        'german': 'Alemán',
        'italian': 'Italiano',
        'portuguese': 'Portugués',
        'russian': 'Ruso',
        'chinese': 'Chino',
        'japanese': 'Japonés',
        'form_instructions_title': '📝 Instrucciones',
        'form_instructions_body': 'Complete el formulario con la información solicitada para obtener una predicción precisa sobre el rendimiento académico.',
        'form_basic_info': '📋 Información Básica del Estudiante',
        'form_school': '🏫 Escuela',
        'form_school_help': 'Escuela del estudiante',
        'school_gp': 'Gabriel Pereira',
        'school_ms': 'Mousinho da Silveira',
        'form_sex': '👤 Sexo',
        'sex_f': 'Femenino',
        'sex_m': 'Masculino',
        'form_age': '🎂 Edad',
        'form_age_help': 'Edad actual del estudiante',
        'form_address': '🏠 Tipo de Dirección',
        'address_u': 'Urbana',
        'address_r': 'Rural',
        'form_medu': '👩‍🎓 Educación de la Madre',
        'edu_0': 'Sin educación',
        'edu_1': 'Educación primaria (4to grado)',
        'edu_2': '5to a 9no grado',
        'edu_3': 'Educación secundaria',
        'edu_4': 'Educación superior',
        'form_fedu': '👨‍🎓 Educación del Padre',
        'form_mjob': '👩‍💼 Trabajo de la Madre',
        'mjob_teacher': 'Profesora',
        'mjob_health': 'Salud',
        'mjob_services': 'Servicios',
        'mjob_at_home': 'En casa',
        'mjob_other': 'Otro',
        'form_reason': '🤔 Razón para elegir la escuela',
        'reason_home': 'Cerca de casa',
        'reason_reputation': 'Reputación de la escuela',
        'reason_course': 'Preferencia del curso',
        'reason_other': 'Otro',
        'form_traveltime': '🚌 Tiempo de viaje a la escuela',
        'traveltime_1': '<15 min',
        'traveltime_2': '15-30 min',
        'traveltime_3': '30 min - 1 hora',
        'traveltime_4': '>1 hora',
        'form_academic_social_info': '📚 Información Académica y Social',
        'form_studytime': '⏱️ Tiempo de estudio semanal',
        'studytime_1': '<2 horas',
        'studytime_2': '2-5 horas',
        'studytime_3': '5-10 horas',
        'studytime_4': '>10 horas',
        'form_failures': '❌ Fallas académicas previas',
        'form_failures_help': 'Número de fallas en clases anteriores',
        'form_higher': '🎓 ¿Quiere seguir educación superior?',
        'yes': 'Sí',
        'no': 'No',
        'form_internet': '🌐 Acceso a Internet en casa',
        'form_dalc': '🍺 Consumo de alcohol entre semana',
        'alc_1': 'Muy bajo',
        'alc_2': 'Bajo',
        'alc_3': 'Medio',
        'alc_4': 'Alto',
        'alc_5': 'Muy alto',
        'form_walc': '🍻 Consumo de alcohol fin de semana',
        'form_health': '🏥 Estado de salud actual',
        'health_1': 'Muy malo',
        'health_2': 'Malo',
        'health_3': 'Regular',
        'health_4': 'Bueno',
        'health_5': 'Muy bueno',
        'form_absences': '🚫 Número de ausencias escolares',
        'form_absences_help': 'Número total de ausencias',
        'form_grades': '📊 Calificaciones',
        'form_g1': '📈 Calificación 1er Período (G1)',
        'form_g1_help': 'Calificación del primer período (0-20)',
        'form_g2': '📈 Calificación 2do Período (G2)',
        'form_g2_help': 'Calificación del segundo período (0-20)',
        'g1_excellent': '✅ G1: Excelente calificación',
        'g1_average': '⚠️ G1: Calificación promedio',
        'g1_low': '❌ G1: Calificación baja',
        'g2_excellent': '✅ G2: Excelente calificación',
        'g2_average': '⚠️ G2: Calificación promedio',
        'g2_low': '❌ G2: Calificación baja',
        'form_ready': '✨ ¿Listo para conocer el resultado?',
        'predict_button_final': '🔮 Predecir Probabilidad de Ingreso',
        'pdf_button_report': '📄 Generar Reporte PDF',
        'factor_higher_yes': '✅ Interés en educación superior',
        'factor_failures_no': '✅ Sin fallas académicas previas',
        'factor_failures_many': '❌ Múltiples fallas académicas',
        'factor_good_studytime': '✅ Buen tiempo de estudio',
        'factor_internet_yes': '✅ Acceso a Internet'
    },
    'en': {
        'main_title': "🎓 University Admission Predictor",
        'subtitle': "**Prediction system based on Machine Learning**",
        'student_info': "📋 Student Information",
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
        'pdf_generation': "📄 PDF Generation",
        'pdf_success': "✅ PDF generated successfully!",
        'pdf_error': "❌ Error generating PDF: {e}",
        'data_summary': "📋 Data Summary",
        'sidebar_info': "ℹ️ Model Information",
        'sidebar_model': "**Model:** {model}\n\n**R² Score:** 86.31%\n**Accuracy:** 91.97%\n**Features:** 19 variables\n\n**Best model trained with real data**",
        'sidebar_report': "📄 Download Training Report",
        'footer': "---",
        'language': "Language",
        'spanish': "Spanish",
        'english': "English",
        'pdf_download': "Download PDF",
        'value': 'Value',
        'french': 'French',
        'german': 'German',
        'italian': 'Italian',
        'portuguese': 'Portuguese',
        'russian': 'Russian',
        'chinese': 'Chinese',
        'japanese': 'Japanese',
        'form_instructions_title': '📝 Instructions',
        'form_instructions_body': 'Complete the form with the requested information to get an accurate prediction of academic performance.',
        'form_basic_info': '📋 Basic Student Information',
        'form_school': '🏫 School',
        'form_school_help': 'Student\'s school',
        'school_gp': 'Gabriel Pereira',
        'school_ms': 'Mousinho da Silveira',
        'form_sex': '👤 Sex',
        'sex_f': 'Female',
        'sex_m': 'Male',
        'form_age': '🎂 Age',
        'form_age_help': 'Current age of the student',
        'form_address': '🏠 Address Type',
        'address_u': 'Urban',
        'address_r': 'Rural',
        'form_medu': '👩‍🎓 Mother\'s Education',
        'edu_0': 'None',
        'edu_1': 'Primary education (4th grade)',
        'edu_2': '5th to 9th grade',
        'edu_3': 'Secondary education',
        'edu_4': 'Higher education',
        'form_fedu': '👨‍🎓 Father\'s Education',
        'form_mjob': '👩‍💼 Mother\'s Job',
        'mjob_teacher': 'Teacher',
        'mjob_health': 'Health',
        'mjob_services': 'Services',
        'mjob_at_home': 'At home',
        'mjob_other': 'Other',
        'form_reason': '🤔 Reason for choosing school',
        'reason_home': 'Close to home',
        'reason_reputation': 'School reputation',
        'reason_course': 'Course preference',
        'reason_other': 'Other',
        'form_traveltime': '🚌 Travel time to school',
        'traveltime_1': '<15 min',
        'traveltime_2': '15-30 min',
        'traveltime_3': '30 min - 1 hour',
        'traveltime_4': '>1 hour',
        'form_academic_social_info': '📚 Academic and Social Information',
        'form_studytime': '⏱️ Weekly study time',
        'studytime_1': '<2 hours',
        'studytime_2': '2-5 hours',
        'studytime_3': '5-10 hours',
        'studytime_4': '>10 hours',
        'form_failures': '❌ Past class failures',
        'form_failures_help': 'Number of past class failures',
        'form_higher': '🎓 Wants to pursue higher education?',
        'yes': 'Yes',
        'no': 'No',
        'form_internet': '🌐 Internet access at home',
        'form_dalc': '🍺 Workday alcohol consumption',
        'alc_1': 'Very Low',
        'alc_2': 'Low',
        'alc_3': 'Medium',
        'alc_4': 'High',
        'alc_5': 'Very High',
        'form_walc': '🍻 Weekend alcohol consumption',
        'form_health': '🏥 Current health status',
        'health_1': 'Very bad',
        'health_2': 'Bad',
        'health_3': 'Average',
        'health_4': 'Good',
        'health_5': 'Very good',
        'form_absences': '🚫 Number of school absences',
        'form_absences_help': 'Total number of school absences',
        'form_grades': '📊 Grades',
        'form_g1': '📈 First Period Grade (G1)',
        'form_g1_help': 'Grade for the first period (0-20)',
        'form_g2': '📈 Second Period Grade (G2)',
        'form_g2_help': 'Grade for the second period (0-20)',
        'g1_excellent': '✅ G1: Excellent grade',
        'g1_average': '⚠️ G1: Average grade',
        'g1_low': '❌ G1: Low grade',
        'g2_excellent': '✅ G2: Excellent grade',
        'g2_average': '⚠️ G2: Average grade',
        'g2_low': '❌ G2: Low grade',
        'form_ready': '✨ Ready to see the result?',
        'predict_button_final': '🔮 Predict Admission Probability',
        'pdf_button_report': '📄 Generate PDF Report',
        'factor_higher_yes': '✅ Interest in higher education',
        'factor_failures_no': '✅ No past class failures',
        'factor_failures_many': '❌ Multiple class failures',
        'factor_good_studytime': '✅ Good study time',
        'factor_internet_yes': '✅ Internet access'
    }
}

# Import additional languages
from languages import additional_languages, get_all_supported_languages, get_language_name
from option_translations import (get_translated_schools, get_translated_specialties, 
                                get_translated_application_modes, get_original_value,
                                SCHOOL_TRANSLATIONS, SPECIALTY_TRANSLATIONS, APPLICATION_MODE_TRANSLATIONS)

# Merge additional languages into LANG_DICT
LANG_DICT.update(additional_languages)

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

# Cargar modelo entrenado que predice probabilidad de ingreso
@st.cache_resource
def cargar_modelo():
    """
    Carga el modelo entrenado que predice probabilidad de ingreso (0 o 1)
    basado en las 19 características del student-por.csv
    """
    try:
        # Intentar cargar el modelo ya entrenado
        model = joblib.load('models/best_model.pkl')
        return model, None
    except FileNotFoundError:
        # Si no existe, recrear el modelo siguiendo la misma lógica del train.py
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.utils import shuffle
        
        # Cargar datos del student-por.csv
        df = pd.read_csv('student-por.csv')
        
        # Preparar datos exactamente como en train.py
        columnas_categoricas = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet' , 'romantic', 'famrel']
        df_copy = df.copy()
        label_encoders = {}
        
        for col in columnas_categoricas:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        df = df_copy.copy()
        
        # Crear columna de ingreso como en train.py
        df['ingreso'] = df['G3'].apply(lambda x: 1 if x > 12 else 0)
        
        # Eliminar columnas como en train.py
        df = df.drop(['famsize', 'Pstatus', 'Fjob', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'romantic', 'famrel', 'freetime', 'goout'], axis=1)
        
        # Preparar features y target
        X = df.drop(['ingreso', 'G3'], axis=1)
        y = df['ingreso']
        X, y = shuffle(X, y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=42)
        
        # Entrenar modelo (Random Forest funciona mejor según el train.py)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model, label_encoders

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

def generar_pdf_student(datos, prediccion):
    """
    Genera un PDF con los datos del formulario student-por y la predicción
    """
    
    lang_dict = LANG_DICT[st.session_state['lang']]

    # Crear PDF
    doc, story = create_pdf("prediccion_academica")
    
    # Título principal
    add_title(story, f"🎓 {lang_dict['main_title']}")
    add_spacer(story, 1, 12)
    
    # Información del estudiante
    add_subtitle(story, f"📋 {lang_dict['student_info']}")
    
    # Crear tabla de datos con información relevante
    datos_tabla = [
        [lang_dict['value'], lang_dict['value']],
        [lang_dict['form_school'], lang_dict['school_gp'] if datos['school'] == 'GP' else lang_dict['school_ms']],
        [lang_dict['form_sex'], lang_dict['sex_f'] if datos['sex'] == 'F' else lang_dict['sex_m']],
        [lang_dict['form_age'], str(datos['age'])],
        [lang_dict['form_address'], lang_dict['address_u'] if datos['address'] == 'U' else lang_dict['address_r']],
        [lang_dict['form_medu'], str(datos['Medu'])],
        [lang_dict['form_fedu'], str(datos['Fedu'])],
        [lang_dict['form_mjob'], datos['Mjob']],
        [lang_dict['form_reason'], datos['reason']],
        [lang_dict['form_traveltime'], str(datos['traveltime'])],
        [lang_dict['form_studytime'], str(datos['studytime'])],
        [lang_dict['form_failures'], str(datos['failures'])],
        [lang_dict['form_higher'], lang_dict['yes'] if datos['higher'] == 'yes' else lang_dict['no']],
        [lang_dict['form_internet'], lang_dict['yes'] if datos['internet'] == 'yes' else lang_dict['no']],
        [lang_dict['form_dalc'], str(datos['Dalc'])],
        [lang_dict['form_walc'], str(datos['Walc'])],
        [lang_dict['form_health'], str(datos['health'])],
        [lang_dict['form_absences'], str(datos['absences'])],
        [lang_dict['form_g1'], f"{datos['G1']:.1f}"],
        [lang_dict['form_g2'], f"{datos['G2']:.1f}"]
    ]
    add_spacer(story, 1, 6)
    
    # Crear DataFrame para la tabla
    df_datos = pd.DataFrame(datos_tabla[1:], columns=datos_tabla[0])
    add_table(story, df_datos)
    add_spacer(story, 1, 6)
    
    # Resultado de predicción si existe
    if prediccion is not None:
        add_subtitle(story, f"🔮 {lang_dict['prediction_result']}")
        
        if prediccion >= 0.5:
            resultado = lang_dict['likely_admission']
        else:
            resultado = lang_dict['unlikely_admission']
        
        add_paragraph(story, f"<b>Resultado:</b> {resultado}")
        add_paragraph(story, f"<b>{lang_dict['admission_probability']}:</b> {prediccion:.2%}")
        add_spacer(story, 1, 6)
    
    # Información del modelo
    add_subtitle(story, f"🤖 {lang_dict['sidebar_info']}")
    add_paragraph(story, "<b>Modelo:</b> Random Forest Regressor")
    add_paragraph(story, "<b>Dataset:</b> Student Performance (Portuguese Language)")
    add_paragraph(story, "<b>Características:</b> 19 variables académicas y sociodemográficas")
    add_paragraph(story, "<b>Objetivo:</b> Predecir la probabilidad de ingreso (G3 > 12)")
    add_spacer(story, 1, 6)
    
    # Fecha de generación
    add_spacer(story, 1, 200)
    generation_text = f"<i>Generado el: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</i>"
    add_paragraph(story, generation_text)
    
    # Construir PDF
    build_pdf(doc, story)
    
    return "prediccion_academica.pdf"

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
        <h4 style="margin-top: 0; color: #2c3e50;">{lang_dict['form_instructions_title']}</h4>
        <p style="margin-bottom: 0;">{lang_dict['form_instructions_body']}</p>
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
            <h3 style="margin-top: 0; color: white;">{lang_dict['form_basic_info']}</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 1. School
        school = st.selectbox(
            lang_dict['form_school'],
            ["GP", "MS"],
            format_func=lambda x: lang_dict['school_gp'] if x == "GP" else lang_dict['school_ms'],
            help=lang_dict['form_school_help']
        )
        
        # 2. Sex
        sex = st.selectbox(
            lang_dict['form_sex'],
            ["F", "M"],
            format_func=lambda x: lang_dict['sex_f'] if x == "F" else lang_dict['sex_m']
        )
        
        # 3. Age
        age = st.number_input(
            lang_dict['form_age'],
            min_value=15,
            max_value=22,
            value=16,
            help=lang_dict['form_age_help']
        )
        
    with col2:
        # 4. Address
        address = st.selectbox(
            lang_dict['form_address'],
            ["U", "R"],
            format_func=lambda x: lang_dict['address_u'] if x == "U" else lang_dict['address_r']
        )
        
        # 5. Mother's Education
        Medu = st.selectbox(
            lang_dict['form_medu'],
            [0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: lang_dict['edu_0'],
                1: lang_dict['edu_1'],
                2: lang_dict['edu_2'],
                3: lang_dict['edu_3'],
                4: lang_dict['edu_4']
            }[x]
        )
        
        # 6. Father's Education
        Fedu = st.selectbox(
            lang_dict['form_fedu'],
            [0, 1, 2, 3, 4],
            format_func=lambda x: {
                0: lang_dict['edu_0'],
                1: lang_dict['edu_1'],
                2: lang_dict['edu_2'],
                3: lang_dict['edu_3'],
                4: lang_dict['edu_4']
            }[x]
        )
        
    with col3:
        # 7. Mother's Job
        Mjob = st.selectbox(
            lang_dict['form_mjob'],
            ["teacher", "health", "services", "at_home", "other"],
            format_func=lambda x: {
                "teacher": lang_dict['mjob_teacher'],
                "health": lang_dict['mjob_health'],
                "services": lang_dict['mjob_services'],
                "at_home": lang_dict['mjob_at_home'],
                "other": lang_dict['mjob_other']
            }[x]
        )
        
        # 8. Reason for choosing school
        reason = st.selectbox(
            lang_dict['form_reason'],
            ["home", "reputation", "course", "other"],
            format_func=lambda x: {
                "home": lang_dict['reason_home'],
                "reputation": lang_dict['reason_reputation'],
                "course": lang_dict['reason_course'],
                "other": lang_dict['reason_other']
            }[x]
        )
        
        # 9. Travel time
        traveltime = st.selectbox(
            lang_dict['form_traveltime'],
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: lang_dict['traveltime_1'],
                2: lang_dict['traveltime_2'],
                3: lang_dict['traveltime_3'],
                4: lang_dict['traveltime_4']
            }[x]
        )
    
    # Separador
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Sección 2: Información Académica y Social
    st.markdown(
        f"""
        <div class="section-card fade-in">
            <h3 style="margin-top: 0; color: white;">{lang_dict['form_academic_social_info']}</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # 10. Study time
        studytime = st.selectbox(
            lang_dict['form_studytime'],
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: lang_dict['studytime_1'],
                2: lang_dict['studytime_2'],
                3: lang_dict['studytime_3'],
                4: lang_dict['studytime_4']
            }[x]
        )
        
        # 11. Past class failures
        failures = st.number_input(
            lang_dict['form_failures'],
            min_value=0,
            max_value=4,
            value=0,
            help=lang_dict['form_failures_help']
        )
        
        # 12. Higher education support
        higher = st.selectbox(
            lang_dict['form_higher'],
            ["yes", "no"],
            format_func=lambda x: lang_dict['yes'] if x == "yes" else lang_dict['no']
        )
        
    with col5:
        # 13. Internet access
        internet = st.selectbox(
            lang_dict['form_internet'],
            ["yes", "no"],
            format_func=lambda x: lang_dict['yes'] if x == "yes" else lang_dict['no']
        )
        
        # 14. Workday alcohol consumption
        Dalc = st.selectbox(
            lang_dict['form_dalc'],
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: lang_dict['alc_1'],
                2: lang_dict['alc_2'],
                3: lang_dict['alc_3'],
                4: lang_dict['alc_4'],
                5: lang_dict['alc_5']
            }[x]
        )
        
        # 15. Weekend alcohol consumption
        Walc = st.selectbox(
            lang_dict['form_walc'],
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: lang_dict['alc_1'],
                2: lang_dict['alc_2'],
                3: lang_dict['alc_3'],
                4: lang_dict['alc_4'],
                5: lang_dict['alc_5']
            }[x]
        )
        
    with col6:
        # 16. Health status
        health = st.selectbox(
            lang_dict['form_health'],
            [1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: lang_dict['health_1'],
                2: lang_dict['health_2'],
                3: lang_dict['health_3'],
                4: lang_dict['health_4'],
                5: lang_dict['health_5']
            }[x],
            index=4  # Default to "Bueno"
        )
        
        # 17. Absences
        absences = st.number_input(
            lang_dict['form_absences'],
            min_value=0,
            max_value=93,
            value=0,
            help=lang_dict['form_absences_help']
        )
    
    # Separador
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Sección 3: Calificaciones
    st.markdown(
        f"""
        <div class="section-card fade-in">
            <h3 style="margin-top: 0; color: white;">{lang_dict['form_grades']}</h3>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col7, col8 = st.columns(2)
    
    with col7:
        # 18. G1 - first period grade
        G1 = st.number_input(
            lang_dict['form_g1'],
            min_value=0,
            max_value=20,
            value=0,
            help=lang_dict['form_g1_help']
        )
        
    with col8:
        # 19. G2 - second period grade
        G2 = st.number_input(
            lang_dict['form_g2'],
            min_value=0,
            max_value=20,
            value=0,
            help=lang_dict['form_g2_help']
        )
    
    # Indicadores visuales de las calificaciones
    col_ind1, col_ind2 = st.columns(2)
    with col_ind1:
        if G1 >= 15:
            st.success(lang_dict['g1_excellent'])
        elif G1 >= 10:
            st.warning(lang_dict['g1_average'])
        elif G1 > 0:
            st.error(lang_dict['g1_low'])
            
    with col_ind2:
        if G2 >= 15:
            st.success(lang_dict['g2_excellent'])
        elif G2 >= 10:
            st.warning(lang_dict['g2_average'])
        elif G2 > 0:
            st.error(lang_dict['g2_low'])
    
    # Separador
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # Botones del formulario con mejor diseño
    st.markdown(
        f"""
        <div style="text-align: center; margin: 2rem 0;">
            <h4 style="color: #2c3e50; margin-bottom: 1.5rem;">{lang_dict['form_ready']}</h4>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    col_btn1, col_btn2 = st.columns(2, gap="large")

    with col_btn1:
        predecir = st.form_submit_button(
            lang_dict['predict_button_final'], 
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        generar_pdf = st.form_submit_button(
            lang_dict['pdf_button_report'], 
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
            st.error(f"❌ {lang_dict['error_found']}")
            for error in errores:
                st.error(f"• {error}")
        else:
            st.success(f"✅ {lang_dict['success_data']}")

            # Realizar predicción si se solicitó
            if predecir:
                st.markdown("---")
                st.subheader(f"🔮 {lang_dict['prediction_result']}")

                try:
                    # Preprocesar datos para el modelo
                    caracteristicas = preprocesar_datos_student(datos_student)

                    # Realizar predicción con el modelo
                    probabilidad_ingreso = model.predict(caracteristicas)[0]

                    # Mostrar resultado
                    col_pred1, col_pred2 = st.columns(2)

                    with col_pred1:
                        if probabilidad_ingreso >= 0.5:
                            st.success(lang_dict['likely_admission'])
                            st.balloons()
                        else:
                            st.error(lang_dict['unlikely_admission'])

                    with col_pred2:
                        st.metric(
                            lang_dict['admission_probability'],
                            f"{probabilidad_ingreso:.2%}",
                        )
                    
                    # Mostrar información del modelo
                    st.info(f"🤖 {lang_dict['model_info'].format(model=type(model).__name__)}")
                    
                except Exception as e:
                    st.error(f"❌ {lang_dict['prediction_error'].format(e=e)}")
                    st.info(f"💡 {lang_dict['prediction_hint']}")

                # Mostrar factores influyentes
                st.subheader(f"📈 {lang_dict['factor_analysis']}")

                # Crear análisis básico
                factores = []
                if G1 >= 15 or G2 >= 15:
                    factores.append(lang_dict['g1_excellent'])
                elif G1 >= 10 or G2 >= 10:
                    factores.append(lang_dict['g1_average'])
                else:
                    factores.append(lang_dict['g1_low'])

                if higher == "yes":
                    factores.append(lang_dict['factor_higher_yes'])

                if age >= 15 and age <= 18:
                    factores.append(lang_dict['typical_age'])

                if failures == 0:
                    factores.append(lang_dict['factor_failures_no'])
                elif failures > 2:
                    factores.append(lang_dict['factor_failures_many'])

                if studytime >= 3:
                    factores.append(lang_dict['factor_good_studytime'])
                    
                if internet == "yes":
                    factores.append(lang_dict['factor_internet_yes'])

                for factor in factores:
                    st.write(factor)

            # Generar PDF si se solicitó
            if generar_pdf:
                st.markdown("---")
                st.subheader(f"📄 {lang_dict['pdf_generation']}")
                
                try:
                    # Usar la predicción ya calculada
                    caracteristicas = preprocesar_datos_student(datos_student)
                    probabilidad_ingreso = model.predict(caracteristicas)[0]
                    
                    # Generar PDF
                    pdf_path = generar_pdf_student(datos_student, probabilidad_ingreso)
                    
                    # Crear nombre de archivo
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"prediccion_academica_{timestamp}.pdf"
                    
                    # Mostrar enlace de descarga
                    st.success(f"✅ {lang_dict['pdf_success']}")
                    st.markdown(get_pdf_download_link(pdf_path, filename), unsafe_allow_html=True)
                    
                    # Limpiar archivo temporal
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ {lang_dict['pdf_error'].format(e=e)}")

            # Mostrar resumen de datos
            st.markdown("---")
            st.subheader(f"📋 {lang_dict['data_summary']}")

            df_resumen = pd.DataFrame.from_records(
                [
                    [lang_dict['form_school'], lang_dict['school_gp'] if school == "GP" else lang_dict['school_ms']],
                    [lang_dict['form_sex'], lang_dict['sex_f'] if sex == "F" else lang_dict['sex_m']],
                    [lang_dict['form_age'], str(age)],
                    [lang_dict['form_address'], lang_dict['address_u'] if address == "U" else lang_dict['address_r']],
                    [lang_dict['form_medu'], str(Medu)],
                    [lang_dict['form_fedu'], str(Fedu)],
                    [lang_dict['form_mjob'], Mjob],
                    [lang_dict['form_reason'], reason],
                    [lang_dict['form_traveltime'], str(traveltime)],
                    [lang_dict['form_studytime'], str(studytime)],
                    [lang_dict['form_failures'], str(failures)],
                    [lang_dict['form_higher'], lang_dict['yes'] if higher == "yes" else lang_dict['no']],
                    [lang_dict['form_internet'], lang_dict['yes'] if internet == "yes" else lang_dict['no']],
                    [lang_dict['form_dalc'], str(Dalc)],
                    [lang_dict['form_walc'], str(Walc)],
                    [lang_dict['form_health'], str(health)],
                    [lang_dict['form_absences'], str(absences)],
                    [lang_dict['form_g1'], f"{G1:.1f}"],
                    [lang_dict['form_g2'], f"{G2:.1f}"]
                ],
                columns=(lang_dict['value'], lang_dict['value'])
            )
            
            try:
                # Use table instead of dataframe to avoid PyArrow issues
                st.table(df_resumen)
            except Exception as e:
                # Fallback display if table fails
                st.write(f"**{lang_dict['data_summary']}**: ")
                for _, row in df_resumen.iterrows():
                    st.write(f"• **{row[lang_dict['value']]}:** {row[lang_dict['value']]}")

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
