import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from make_pdf import create_pdf, build_pdf, add_title, add_subtitle, add_paragraph, add_spacer, add_list, add_image, add_table

ad_df = pd.read_csv("Datos_abiertos_admision_2021_1_2024_1.csv")


ad_df = ad_df.drop(['IDHASH', 'COLEGIO_DEPA', 'COLEGIO_PROV', 'COLEGIO_DIST',
                    'COLEGIO_PAIS', 'DOMICILIO_DEPA', 'DOMICILIO_PROV',
                    'DOMICILIO_DIST', 'NACIMIENTO_PAIS', 'NACIMIENTO_DEPA',
                    'NACIMIENTO_PROV', 'NACIMIENTO_DIST'], axis=1)

print("Estadísticas descriptivas:")
estadisticas = ad_df.describe()
print(estadisticas)

# Eliminar filas con valores nulos
ad_df = ad_df.dropna()

# Categorización
columnas_categoricas = ['COLEGIO', 'ESPECIALIDAD', 'SEXO', 'MODALIDAD', 'INGRESO']
label_encoders = {}  # Diccionario para guardar los encoders
for col in columnas_categoricas:
  le = LabelEncoder()
  col_name = col + "_ENCODED"
  ad_df[col_name] = le.fit_transform(ad_df[col])
  label_encoders[col] = le  # Guardar el encoder
  ad_df = ad_df.drop(col, axis=1)

# Visualización de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(ad_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.tight_layout()
plt.savefig("images/matriz_correlacion.png")
#plt.show()

train_size = 0.8
test_size = 1 - train_size

# Preparación de datos
X = ad_df.drop(['INGRESO_ENCODED'], axis=1)
y = ad_df['INGRESO_ENCODED']
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Estandarización
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

print("\n" + "="*60)
print("INFORMACIÓN DE ENTRENAMIENTO")
print("="*60)

print("\nDatos preparados - Shape del conjunto de entrenamiento:", X_train_scaled.shape)
print("Datos preparados - Shape del conjunto de prueba:", X_test_scaled.shape)

train_length = X_train_scaled.shape[0]
test_length = X_test_scaled.shape[0]

# Diccionario para almacenar modelos y resultados
models = {}
results = {}

print("\n" + "="*60)
print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("="*60)

# 1. REGLRESIÓN LINEA
print("\n1. Regresión Lineal:")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

models['Regresión Lineal'] = lr
results['Regresión Lineal'] = {
    'y_pred': lr_pred,
    'r2': r2_score(y_test, lr_pred),
    'mse': mean_squared_error(y_test, lr_pred),
    'mae': mean_absolute_error(y_test, lr_pred)
}

print(f"R² Score: {results['Regresión Lineal']['r2']:.4f}")
print(f"MSE: {results['Regresión Lineal']['mse']:.4f}")
print(f"MAE: {results['Regresión Lineal']['mae']:.4f}")

# 2. Bosques Aleatorios
print("\n2. Bosques Aleatorios:")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Bosques Aleatorios no necesita estandarización
rf_pred = rf.predict(X_test)

models['Bosques Aleatorios'] = rf
results['Bosques Aleatorios'] = {
    'y_pred': rf_pred,
    'r2': r2_score(y_test, rf_pred),
    'mse': mean_squared_error(y_test, rf_pred),
    'mae': mean_absolute_error(y_test, rf_pred)
}

print(f"R² Score: {results['Bosques Aleatorios']['r2']:.4f}")
print(f"MSE: {results['Bosques Aleatorios']['mse']:.4f}")
print(f"MAE: {results['Bosques Aleatorios']['mae']:.4f}")

# 3. SUPPORT VECTOR REGRESSION
print("\n3. Support Vector Regression:")
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr.fit(X_train_scaled, y_train)
svr_pred = svr.predict(X_test_scaled)

models['Regresión de Vectores de Soporte'] = svr
results['Regresión de Vectores de Soporte'] = {
    'y_pred': svr_pred,
    'r2': r2_score(y_test, svr_pred),
    'mse': mean_squared_error(y_test, svr_pred),
    'mae': mean_absolute_error(y_test, svr_pred)
}

print(f"R² Score: {results['Regresión de Vectores de Soporte']['r2']:.4f}")
print(f"MSE: {results['Regresión de Vectores de Soporte']['mse']:.4f}")
print(f"MAE: {results['Regresión de Vectores de Soporte']['mae']:.4f}")

# 4. Potenciación de Gradiente
print("\n4. Potenciación de Gradiente:")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)  # No necesita estandarización
gb_pred = gb.predict(X_test)

models['Potenciación de Gradiente'] = gb
results['Potenciación de Gradiente'] = {
    'y_pred': gb_pred,
    'r2': r2_score(y_test, gb_pred),
    'mse': mean_squared_error(y_test, gb_pred),
    'mae': mean_absolute_error(y_test, gb_pred)
}

print(f"R² Score: {results['Potenciación de Gradiente']['r2']:.4f}")
print(f"MSE: {results['Potenciación de Gradiente']['mse']:.4f}")
print(f"MAE: {results['Potenciación de Gradiente']['mae']:.4f}")

# TABLA COMPARATIVA DE RESULTADOS
print("\n" + "="*60)
print("COMPARACIÓN DE RESULTADOS")
print("="*60)

comparison_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'R² Score': [results[model]['r2'] for model in results.keys()],
    'MSE': [results[model]['mse'] for model in results.keys()],
    'MAE': [results[model]['mae'] for model in results.keys()]
})

comparison_df = comparison_df.sort_values('R² Score', ascending=False).round(4)
print(comparison_df)

# VISUALIZACIONES COMPARATIVAS

# 1. Gráfico de barras para R² Score
plt.figure(figsize=(5, 6))
plt.bar(comparison_df['Modelo'], comparison_df['R² Score'], color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación R² Score')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparacion_r2.png")
#plt.show()

# 2. Gráfico de barras para MSE
plt.figure(figsize=(5, 6))
plt.bar(comparison_df['Modelo'], comparison_df['MSE'], color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación Error Cuadrado Promedio (menor es mejor)')
plt.ylabel('ECP')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparacion_mse.png")
#plt.show()

# 3. Matrices de confusión
colors = ['blue', 'green', 'red', 'orange']
for i, (model_name, color) in enumerate(zip(results.keys(), colors)):
  y_pred = [1 if i > 0.5 else 0  for i in results[model_name]['y_pred']]
  y_test_bin = [1 if i > 0.5 else 0 for i in y_test]
  cm = confusion_matrix(y_test_bin, y_pred)
  tn, fp, fn, tp = cm.ravel()
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp + fn) > 0 else 0
  f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  mcc = matthews_corrcoef(y_test_bin, y_pred)
  results[model_name]['accuracy'] = accuracy
  results[model_name]['precision'] = precision
  results[model_name]['recall'] = recall
  results[model_name]['f1'] = f1
  results[model_name]['mcc'] = mcc
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap="Blues")
  plt.tight_layout()
  plt.savefig(f"images/matriz_confusion_{model_name}.png")
  #plt.show()

clasificacion_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Precisión': [results[model]['precision'] for model in results.keys()],
    'Sensibilidad': [results[model]['recall'] for model in results.keys()],
    'Exactitud': [results[model]['accuracy'] for model in results.keys()],
    'F1': [results[model]['f1'] for model in results.keys()],
    'R2': [results[model]['r2'] for model in results.keys()],
    'EAP': [results[model]['mae'] for model in results.keys()],
    'ECP': [results[model]['mse'] for model in results.keys()],
    'CM': [results[model]['mcc'] for model in results.keys()],
})

clasificacion_df = clasificacion_df.sort_values('Precisión', ascending=False).round(4)
print(clasificacion_df)

# 4. Gráfico de líneas para todas las métricas normalizadas
plt.figure(figsize=(5, 6))
metrics_norm = comparison_df.copy()
metrics_norm['R² Score'] = metrics_norm['R² Score'] / metrics_norm['R² Score'].max()
metrics_norm['MSE_inv'] = 1 - (metrics_norm['MSE'] / metrics_norm['MSE'].max())  # Invertir MSE para que mayor sea mejor
metrics_norm['MAE_inv'] = 1 - (metrics_norm['MAE'] / metrics_norm['MAE'].max())  # Invertir MAE para que mayor sea mejor

x = range(len(metrics_norm))
plt.plot(x, metrics_norm['R² Score'], 'o-', label='R² Score (norm)', linewidth=2)
plt.plot(x, metrics_norm['MSE_inv'], 's-', label='MSE (inv norm)', linewidth=2)
plt.plot(x, metrics_norm['MAE_inv'], '^-', label='MAE (inv norm)', linewidth=2)
plt.xticks(x, metrics_norm['Modelo'], rotation=45)
plt.ylabel('Puntuación Normalizada')
plt.title('Métricas Normalizadas')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("images/comparacion_metricas.png")
#plt.show()

# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS (para modelos que lo soportan)
print("\n" + "="*60)
print("IMPORTANCIA DE CARACTERÍSTICAS")
print("="*60)

# Random Forest - Importancia de características
print("\nRandom Forest - Importancia de características:")
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print(rf_importance)

# Gradient Boosting - Importancia de características
print("\nGradient Boosting - Importancia de características:")
gb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)
print(gb_importance)

# Regresión Lineal - Coeficientes
print("\nRegresión Lineal - Coeficientes:")
lr_coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(lr_coef)

# Visualización de importancia de características
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.barh(rf_importance['Feature'], rf_importance['Importance'])
plt.title('Random Forest - Importancia')
plt.xlabel('Importancia')

plt.subplot(1, 3, 2)
plt.barh(gb_importance['Feature'], gb_importance['Importance'])
plt.title('Gradient Boosting - Importancia')
plt.xlabel('Importancia')

plt.subplot(1, 3, 3)
plt.barh(lr_coef['Feature'], lr_coef['Coefficient'])
plt.title('Regresión Lineal - Coeficientes')
plt.xlabel('Coeficiente')

plt.tight_layout()
#plt.show()

# MEJOR MODELO
best_model_name = comparison_df.iloc[0]['Modelo']
best_r2 = comparison_df.iloc[0]['R² Score']

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"R² Score: {best_r2:.4f}")
print(f"Esto significa que el modelo explica el {best_r2*100:.2f}% de la varianza en los datos.")

# GUARDAR MODELOS Y SCALERS
print("\n" + "="*60)
print("GUARDANDO MODELOS Y SCALERS")
print("="*60)

# Crear directorio para modelos si no existe
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Directorio '{models_dir}' creado.")

# Guardar todos los modelos entrenados
for model_name, model in models.items():
    model_filename = f"{models_dir}/{model_name.replace(' ', '_').lower()}.pkl"
    with open(model_filename, 'wb') as f:
        joblib.dump(model, f)
    print(f"Modelo '{model_name}' guardado en: {model_filename}")

# Guardar el StandardScaler
scaler_filename = f"{models_dir}/standard_scaler.pkl"
with open(scaler_filename, 'wb') as f:
    joblib.dump(ss, f)
print(f"StandardScaler guardado en: {scaler_filename}")

# Guardar los LabelEncoders
encoders_filename = f"{models_dir}/label_encoders.pkl"
with open(encoders_filename, 'wb') as f:
    joblib.dump(label_encoders, f)
print(f"LabelEncoders guardados en: {encoders_filename}")

# Guardar información de las columnas para preprocesamiento futuro
column_info = {
    'feature_columns': list(X.columns),
    'categorical_columns': columnas_categoricas,
    'target_column': 'INGRESO_ENCODED'
}

column_info_filename = f"{models_dir}/column_info.pkl"
with open(column_info_filename, 'wb') as f:
    joblib.dump(column_info, f)
print(f"Información de columnas guardada en: {column_info_filename}")

# Guardar el mejor modelo por separado para fácil acceso
best_model = models[best_model_name]
best_model_filename = f"{models_dir}/best_model.pkl"
with open(best_model_filename, 'wb') as f:
    joblib.dump(best_model, f)
print(f"Mejor modelo '{best_model_name}' guardado en: {best_model_filename}")

print(f"\n✅ Todos los modelos y scalers han sido guardados en el directorio '{models_dir}/'")
print("Archivos guardados:")
print(f"  - Modelos individuales: {len(models)} archivos .pkl")
print(f"  - StandardScaler: standard_scaler.pkl")
print(f"  - LabelEncoders: label_encoders.pkl")
print(f"  - Información de columnas: column_info.pkl")
print(f"  - Mejor modelo: best_model.pkl")

doc, story = create_pdf("reporte")

# Titulo
add_title(story, "Reporte de Modelos de Clasificación")
add_spacer(story, 1,12)

# Dataset
#ad_df_short = ad_df.head()
#add_subtitle(story, "Dataset")
#add_table(story, ad_df_short)
#add_spacer(story, 1,12)

# Modelos usados
models_list = ['Regresión Lineal',
          'Bosques Aleatorios',
          'Regresión de Vectores de Soporte',
          'Potenciación de Gradiente']
add_subtitle(story, "Modelos a comparar")
add_list(story, models_list)
add_spacer(story, 1,6)

# Matriz de correlacion
add_subtitle(story, "Matriz de correlación")
add_image(story, "images/matriz_correlacion.png", 500 , 400)
add_spacer(story, 1,6)

# Datos de entrenamiento y prueba
add_subtitle(story, "Datos de entrenamiento y prueba")
add_paragraph(story, f"Tamaño de conjunto de datos de entrenamiento: {train_length} ({train_size*100:.2f}%)")
add_paragraph(story, f"Tamaño de conjunto de datos de prueba: {test_length} ({test_size*100:.2f}%)")
add_spacer(story, 1,6)

# Comparación de modelos
add_subtitle(story, "Comparación R2")
add_image(story, "images/comparacion_r2.png", 240, 300)
add_subtitle(story, "Comparación Error Cuadrado Promedio")
add_image(story, "images/comparacion_mse.png", 240, 300)
add_subtitle(story, "Comparación Métricas")
add_image(story, "images/comparacion_metricas.png", 240, 300)

# Matrices de confusión
for model in models_list:
  add_subtitle(story, f"Matriz de confusión de {model}")
  add_image(story, f"images/matriz_confusion_{model}.png", 320, 240)
add_spacer(story, 1,6)

# Análisis de clasificación
add_subtitle(story, "Análisis de clasificación")
add_table(story, clasificacion_df)
add_spacer(story, 1,6)

# McNemar test between all pairs of models
model_names_sorted = list(comparison_df['Modelo'])
mcnemar_matrix_p = pd.DataFrame(index=model_names_sorted, columns=model_names_sorted)
mcnemar_matrix_stat = pd.DataFrame(index=model_names_sorted, columns=model_names_sorted)

# Binarize y_test once
y_test_bin = [1 if i > 0.5 else 0 for i in y_test]

# Store binarized predictions for each model
binarized_preds = {model: [1 if i > 0.5 else 0 for i in results[model]['y_pred']] for model in model_names_sorted}

for i, model_i in enumerate(model_names_sorted):
    for j, model_j in enumerate(model_names_sorted):
        if i == j:
            mcnemar_matrix_p.loc[model_i, model_j] = '-'
            mcnemar_matrix_stat.loc[model_i, model_j] = '-'
        elif pd.isnull(mcnemar_matrix_p.loc[model_i, model_j]):
            y_pred_i = binarized_preds[model_i]
            y_pred_j = binarized_preds[model_j]
            contingency = np.zeros((2,2), dtype=int)
            for yt, yi, yj in zip(y_test_bin, y_pred_i, y_pred_j):
                if yi == yt and yj == yt:
                    contingency[0,0] += 1  # both correct
                elif yi == yt and yj != yt:
                    contingency[0,1] += 1  # i correct only
                elif yi != yt and yj == yt:
                    contingency[1,0] += 1  # j correct only
                else:
                    contingency[1,1] += 1  # both wrong
            mcnemar_result = mcnemar(contingency, exact=True)
            mcnemar_matrix_p.loc[model_i, model_j] = mcnemar_result.pvalue
            mcnemar_matrix_stat.loc[model_i, model_j] = mcnemar_result.statistic
            # Fill symmetric value
            mcnemar_matrix_p.loc[model_j, model_i] = mcnemar_result.pvalue
            mcnemar_matrix_stat.loc[model_j, model_i] = mcnemar_result.statistic

# Add McNemar summary tables to PDF
add_subtitle(story, "Matriz de p-valores de McNemar entre modelos")
add_table(story, mcnemar_matrix_p.round(4))
add_spacer(story, 1,6)

add_subtitle(story, "Matriz de estadísticos de McNemar entre modelos")
add_table(story, mcnemar_matrix_stat.round(2))
add_spacer(story, 1,6)

# Mejor modelo
add_subtitle(story, "Modelo Optimo")
add_paragraph(story, f"<b>MEJOR MODELO:</b> {best_model_name}")
add_paragraph(story, f"• Precisión: {results[best_model_name]['precision']}")
add_paragraph(story, f"• Sensibilidad: {results[best_model_name]['recall']}")
add_paragraph(story, f"• Puntuación F1: {results[best_model_name]['f1']}")
add_paragraph(story, f"• Exactitud: {results[best_model_name]['accuracy']}")
add_paragraph(story, f"• Coeficiente R2: {results[best_model_name]['r2']}")
add_paragraph(story, f"• Error Absoluto Promedio: {results[best_model_name]['mae']}")
add_paragraph(story, f"• Error Cuadrado Promedio: {results[best_model_name]['mse']}")
add_paragraph(story, f"• Coeficiente de Mathews: {results[best_model_name]['mcc']}")

# McNemar test between best and second-best model
model_names_sorted = list(comparison_df['Modelo'])
best_model_name = model_names_sorted[0]
second_best_model_name = model_names_sorted[1]
y_pred_best = [1 if i > 0.5 else 0 for i in results[best_model_name]['y_pred']]
y_pred_second = [1 if i > 0.5 else 0 for i in results[second_best_model_name]['y_pred']]
y_test_bin = [1 if i > 0.5 else 0 for i in y_test]
# Build 2x2 contingency table
# Rows: y_test, Cols: predictions
# Table: [[both correct, best correct only], [second correct only, both wrong]]
contingency = np.zeros((2,2), dtype=int)
for yt, yb, ys in zip(y_test_bin, y_pred_best, y_pred_second):
    if yb == yt and ys == yt:
        contingency[0,0] += 1  # both correct
    elif yb == yt and ys != yt:
        contingency[0,1] += 1  # best correct only
    elif yb != yt and ys == yt:
        contingency[1,0] += 1  # second correct only
    else:
        contingency[1,1] += 1  # both wrong
mcnemar_result = mcnemar(contingency, exact=True)
# Add to results for reporting
results[best_model_name]['mcnemar_pvalue_vs_second'] = mcnemar_result.pvalue
results[best_model_name]['mcnemar_statistic_vs_second'] = mcnemar_result.statistic

add_paragraph(story, f"• McNemar p-valor (vs {second_best_model_name}): {results[best_model_name]['mcnemar_pvalue_vs_second']}")
add_paragraph(story, f"• McNemar estadístico (vs {second_best_model_name}): {results[best_model_name]['mcnemar_statistic_vs_second']}")

# Generar PDF
build_pdf(doc, story)
print("\n✅ PDF generado correctamente")

def run_streamlit():
    os.system('streamlit run app.py --server.port 8501 --server.headless true -server.fileWatcherType none --browser.gatherUsageStats false')