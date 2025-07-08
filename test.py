import joblib
import pandas as pd

model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/standard_scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')

categorical_columns = ['COLEGIO', 'ESPECIALIDAD', 'SEXO', 'MODALIDAD']

test_data = {
    'año_nacimiento': 2000,
    'sexo': 'MASCULINO',
    'colegio': 'LA DIVINA PROVIDENCIA',
    'año_egreso': 2018,
    'especialidad': 'INGENIERÍA DE SISTEMAS',
    'año_postulacion': 2024,
    'ciclo': 'I Ciclo',
    'modalidad': 'ORDINARIO',
    'calificacion_final': 15.5
}

df = pd.DataFrame({
    'COLEGIO_ANIO_EGRESO': [test_data['año_egreso']],
    'ANIO_POSTULA': [test_data['año_postulacion']],
    'CICLO_POSTULA': [1 if test_data['ciclo'] == 'I Ciclo' else 2],
    'ANIO_NACIMIENTO': [test_data['año_nacimiento']],
    'CALIF_FINAL': [test_data['calificacion_final']],
    'COLEGIO': [test_data['colegio']],
    'ESPECIALIDAD': [test_data['especialidad']],
    'SEXO': [test_data['sexo']],
    'MODALIDAD': [test_data['modalidad']],
})

for col in categorical_columns:
    le = label_encoders[col]
    df[col + "_ENCODED"] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    df = df.drop(col, axis=1)

prediction = model.predict(df)
print(prediction)