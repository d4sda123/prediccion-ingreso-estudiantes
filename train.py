from re import S
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, brier_score_loss
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from make_pdf import create_pdf, build_pdf, add_title, add_subtitle, add_paragraph, add_spacer, add_list, add_image, add_table
from scipy.stats import shapiro, kstest, f_oneway, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# Lectura de datos
df = pd.read_csv("student-por.csv")

# Plots de distribución de frecuencia para TIEMPOO_ESTUDIO y FALLAS
plt.figure(figsize=(10, 6))
df['studytime'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribución de Frecuencia de TIEMPO_ESTUDIO')
plt.xlabel('TIEMPO_ESTUDIO')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/frecuencia_estudio.png')
plt.close()

plt.figure(figsize=(10, 6))
df['failures'].value_counts().plot(kind='bar', color='lightgreen')
plt.title('Distribución de Frecuencia de FALLAS')
plt.xlabel('FALLAS')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('images/frecuencia_fallas.png')
plt.close()

# Gráfico de dispersión para columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols])
    plt.suptitle('Matriz de Dispersión de Variables Numéricas', y=1.02)
    plt.savefig('images/dispersion_numericas.png')
    plt.close()

# Plots de medidas de tendencia central para columnas numéricas
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col].to_numpy(), kde=True, color='lightblue', bins=30)
    mean = float(df[col].mean())
    median = float(df[col].median())
    mode = float(df[col].mode().iloc[0]) if not df[col].mode().empty else np.nan
    plt.axvline(mean, color='red', linestyle='--', label=f'Media: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='-.', label=f'Mediana: {median:.2f}')
    plt.axvline(mode, color='blue', linestyle=':', label=f'Moda: {mode:.2f}')
    plt.title(f'Medidas de tendencia central - {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/medidas_{col}.png')
    plt.close()

print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60+"\n")
estadisticas = df.describe()
print(estadisticas)


# Categorización de datos
columnas_categoricas = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet' , 'romantic', 'famrel']
df_copy = df.copy()
label_encoders = {}
for col in columnas_categoricas:
  le = LabelEncoder()
  col_name = col + "_ENCODED"
  df_copy[col] = le.fit_transform(df[col])

df = df_copy.copy()

df['ingreso'] = df['G3'].apply(lambda x: 1 if x > 12 else 0)

# Visualización de correlaciones
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0)
plt.tight_layout()
plt.savefig("images/matriz_correlacion.png")

df = df.drop(['famsize', 'Pstatus', 'Fjob', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'romantic', 'famrel', 'freetime', 'goout'], axis=1)

train_size = 0.8
test_size = 1 - train_size

# Preparación de datos
X = df.drop(['ingreso', 'G3'], axis=1)
y = df['ingreso']
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

# 1. Regresión Lineal
print("\n1. Regresión Lineal:")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

models['Regresión Lineal'] = lr
results['Regresión Lineal'] = {
    'y_real': y_test,
    'y_pred': lr_pred,
    'r2': r2_score(y_test, lr_pred),
    'mse': mean_squared_error(y_test, lr_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
    'mae': mean_absolute_error(y_test, lr_pred),
    'brier': brier_score_loss(y_test, np.clip(lr_pred, 0, 1))
}

print(f"R² Score: {results['Regresión Lineal']['r2']:.4f}")
print(f"MSE: {results['Regresión Lineal']['mse']:.4f}")
print(f"RMSE: {results['Regresión Lineal']['rmse']:.4f}")
print(f"MAE: {results['Regresión Lineal']['mae']:.4f}")
print(f"Brier Score: {results['Regresión Lineal']['brier']:.4f}")

# 2. Bosques Aleatorios
print("\n2. Bosques Aleatorios:")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Bosques Aleatorios no necesita estandarización
rf_pred = rf.predict(X_test)

models['Bosques Aleatorios'] = rf
results['Bosques Aleatorios'] = {
    'y_real': y_test,
    'y_pred': rf_pred,
    'r2': r2_score(y_test, rf_pred),
    'mse': mean_squared_error(y_test, rf_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'mae': mean_absolute_error(y_test, rf_pred),
    'brier': brier_score_loss(y_test, np.clip(rf_pred, 0, 1))   
}

print(f"R² Score: {results['Bosques Aleatorios']['r2']:.4f}")
print(f"MSE: {results['Bosques Aleatorios']['mse']:.4f}")
print(f"RMSE: {results['Bosques Aleatorios']['rmse']:.4f}")
print(f"MAE: {results['Bosques Aleatorios']['mae']:.4f}")
print(f"Brier Score: {results['Bosques Aleatorios']['brier']:.4f}")

# 3. Regresión de Vectores de Soporte
print("\n3. Support Vector Regression:")
svr = SVR(kernel='rbf', C=1.0, gamma='scale')
svr.fit(X_train_scaled, y_train)
svr_pred = svr.predict(X_test_scaled)

models['Regresión de Vectores de Soporte'] = svr
results['Regresión de Vectores de Soporte'] = {
    'y_real': y_test,
    'y_pred': svr_pred,
    'r2': r2_score(y_test, svr_pred),
    'mse': mean_squared_error(y_test, svr_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, svr_pred)),
    'mae': mean_absolute_error(y_test, svr_pred),
    'brier': brier_score_loss(y_test, np.clip(svr_pred, 0, 1))
}

print(f"R² Score: {results['Regresión de Vectores de Soporte']['r2']:.4f}")
print(f"MSE: {results['Regresión de Vectores de Soporte']['mse']:.4f}")
print(f"RMSE: {results['Regresión de Vectores de Soporte']['rmse']:.4f}")
print(f"MAE: {results['Regresión de Vectores de Soporte']['mae']:.4f}")
print(f"Brier Score: {results['Regresión de Vectores de Soporte']['brier']:.4f}")

# 4. Potenciación de Gradiente
print("\n4. Potenciación de Gradiente:")
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

models['Potenciación de Gradiente'] = gb
results['Potenciación de Gradiente'] = {
    'y_real': y_test,
    'y_pred': gb_pred,
    'r2': r2_score(y_test, gb_pred),
    'mse': mean_squared_error(y_test, gb_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
    'mae': mean_absolute_error(y_test, gb_pred),
    'brier': brier_score_loss(y_test, np.clip(gb_pred, 0, 1))
}

print(f"R² Score: {results['Potenciación de Gradiente']['r2']:.4f}")
print(f"MSE: {results['Potenciación de Gradiente']['mse']:.4f}")
print(f"RMSE: {results['Potenciación de Gradiente']['rmse']:.4f}")
print(f"MAE: {results['Potenciación de Gradiente']['mae']:.4f}")
print(f"Brier Score: {results['Potenciación de Gradiente']['brier']:.4f}")

# 5. XGBoost
print("\n5. XGBoost:")
xgb = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

models['XGBoost'] = xgb
results['XGBoost'] = {
    'y_real': y_test,
    'y_pred': xgb_pred,
    'r2': r2_score(y_test, xgb_pred),
    'mse': mean_squared_error(y_test, xgb_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'mae': mean_absolute_error(y_test, xgb_pred),
    'brier': brier_score_loss(y_test, np.clip(xgb_pred, 0, 1))
}

print(f"R² Score: {results['XGBoost']['r2']:.4f}")
print(f"MSE: {results['XGBoost']['mse']:.4f}")
print(f"RMSE: {results['XGBoost']['rmse']:.4f}")
print(f"MAE: {results['XGBoost']['mae']:.4f}")
print(f"Brier Score: {results['XGBoost']['brier']:.4f}")

# 6. LightGBM
print("\n6. LightGBM:")
lgbm = LGBMRegressor(n_estimators=100, random_state=42)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)

models['LightGBM'] = lgbm
results['LightGBM'] = {
    'y_real': y_test,
    'y_pred': lgbm_pred,
    'r2': r2_score(y_test, lgbm_pred),
    'mse': mean_squared_error(y_test, lgbm_pred),
    'rmse': np.sqrt(mean_squared_error(y_test, lgbm_pred)),
    'mae': mean_absolute_error(y_test, lgbm_pred),
    'brier': brier_score_loss(y_test, np.clip(lgbm_pred, 0, 1))
}

print(f"R² Score: {results['LightGBM']['r2']:.4f}")
print(f"MSE: {results['LightGBM']['mse']:.4f}")
print(f"RMSE: {results['LightGBM']['rmse']:.4f}")
print(f"MAE: {results['LightGBM']['mae']:.4f}")
print(f"Brier Score: {results['LightGBM']['brier']:.4f}")

# TABLA COMPARATIVA DE RESULTADOS
print("\n" + "="*60)
print("COMPARACIÓN DE RESULTADOS")
print("="*60)

comparison_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'R² Score': [results[model]['r2'] for model in results.keys()],
    'MSE': [results[model]['mse'] for model in results.keys()],
    'RMSE': [results[model]['rmse'] for model in results.keys()],
    'MAE': [results[model]['mae'] for model in results.keys()],
    'Brier': [results[model]['brier'] for model in results.keys()]
})

# Brier Score Comparison Plot
plt.figure(figsize=(5, 6))
brier_scores = comparison_df['Brier']
bars_brier = plt.bar(comparison_df['Modelo'], brier_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación Brier Score (menor es mejor)')
plt.ylabel('Brier Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars_brier, brier_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(brier_scores) * 0.01), f'{value:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("images/comparacion_brier.png")
#plt.show()

comparison_df = comparison_df.sort_values('R² Score', ascending=False).round(4)
print(comparison_df)

# VISUALIZACIONES COMPARATIVAS

# 1. Gráfico de barras para R² Score
plt.figure(figsize=(5, 6))
r2_scores = comparison_df['R² Score'] * 100
bars_r2 = plt.bar(comparison_df['Modelo'], r2_scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación R² Score')
plt.ylabel('R² Score (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars_r2, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("images/comparacion_r2.png")

# 2. Gráfico de barras para MSE
plt.figure(figsize=(5, 6))
mse_vals = comparison_df['MSE']
bars_mse = plt.bar(comparison_df['Modelo'], mse_vals, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación MSE (menor es mejor)')
plt.ylabel('ECP')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars_mse, mse_vals):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(mse_vals) * 0.01), f'{value:.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("images/comparacion_mse.png")

# Remove any code that uses confusion_matrix or ConfusionMatrixDisplay

clasificacion_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'R2': [results[model]['r2'] for model in results.keys()],
    'MAE': [results[model]['mae'] for model in results.keys()],
    'RMCE': [results[model]['rmse'] for model in results.keys()],
    'MCE': [results[model]['mse'] for model in results.keys()],
    'Brier': [results[model]['brier'] for model in results.keys()],
})

clasificacion_df = clasificacion_df.sort_values('R2', ascending=False).round(4)
print(clasificacion_df)

# 3. Gráfico de barras para Precisión (en porcentaje)
plt.figure(figsize=(5, 6))
precisiones = clasificacion_df['R2'] * 100
bars = plt.bar(clasificacion_df['Modelo'], precisiones, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Comparación de Precisión')
plt.ylabel('Precisión (%)')
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, value in zip(bars, precisiones):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig("images/comparacion_precision.png")

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
plt.title('Bosques Aleatorios - Importancia')
plt.xlabel('Importancia')

plt.subplot(1, 3, 2)
plt.barh(gb_importance['Feature'], gb_importance['Importance'])
plt.title('Potenciación de Gradiente - Importancia')
plt.xlabel('Importancia')

plt.subplot(1, 3, 3)
plt.barh(lr_coef['Feature'], lr_coef['Coefficient'])
plt.title('Regresión Lineal - Coeficientes')
plt.xlabel('Coeficiente')
plt.tight_layout()

posthoc_result = None
posthoc_plot_path = None

# CORRECCIÓN DE LA COMPARACIÓN DE MODELOS
# En lugar de usar residuales, usar métricas de rendimiento
# Crear una matriz donde cada fila es una observación y cada columna es un modelo
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS CORREGIDA: ANOVA o FRIEDMAN")
print("="*60)
# Si quieres usar métricas agregadas, necesitas repetir el experimento
# con diferentes splits o usar bootstrap para generar múltiples muestras
# Para una comparación más robusta, usemos bootstrap
from sklearn.utils import resample

n_bootstrap = 51  # Número de muestras bootstrap
bootstrap_scores = {model: [] for model in results.keys()}

print("Realizando bootstrap para obtener múltiples muestras...")
for i in range(n_bootstrap):
    # Crear muestra bootstrap
    X_boot, y_boot = resample(X_test, y_test, random_state=i)
    
    # Evaluar cada modelo en la muestra bootstrap
    for model_name in results.keys():
        model = models[model_name]
        
        if model_name in ['Regresión Lineal', 'Regresión de Vectores de Soporte']:
            # Modelos que necesitan datos escalados
            X_boot_scaled = ss.transform(X_boot)
            y_pred_boot = model.predict(X_boot_scaled)
        else:
            # Modelos que no necesitan escalado
            y_pred_boot = model.predict(X_boot)
        
        # Calcular R²
        r2_boot = r2_score(y_boot, y_pred_boot)
        bootstrap_scores[model_name].append(r2_boot)

# Convertir a matriz para las pruebas estadísticas
bootstrap_matrix = np.array([bootstrap_scores[model] for model in results.keys()]).T
model_names = list(results.keys())

print(f"Matriz bootstrap creada: {bootstrap_matrix.shape}")
print("Estadísticas bootstrap por modelo:")
for i, model in enumerate(model_names):
    mean_score = np.mean(bootstrap_matrix[:, i])
    std_score = np.std(bootstrap_matrix[:, i])
    print(f"  {model}: Media R² = {mean_score} ± {std_score}")

# Prueba de normalidad en las muestras bootstrap
from scipy.stats import shapiro, kstest
print("\nTest de normalidad en muestras bootstrap:")
bootstrap_normal = []
for i, model in enumerate(model_names):
    if len(bootstrap_matrix[:, i]) <= 50:
        stat, p = shapiro(bootstrap_matrix[:, i])
        test = 'Shapiro-Wilk'
    else:
        stat, p = kstest(bootstrap_matrix[:, i], 'norm')
        test = 'Kolmogorov-Smirnov'
    
    bootstrap_normal.append(p > 0.05)
    print(f"  {model} - {test}: stat={stat}, p={p}")

all_bootstrap_normal = all(bootstrap_normal)
print(f"\nTodas las muestras bootstrap son normales: {all_bootstrap_normal}")

# Aplicar el test apropiado
if all_bootstrap_normal:
    # ANOVA si todas las muestras son normales
    from scipy.stats import f_oneway
    anova_stat, anova_p = f_oneway(*[bootstrap_matrix[:, i] for i in range(len(model_names))])
    print(f"\nANOVA en muestras bootstrap:")
    print(f"  F-statistic: {anova_stat}")
    print(f"  p-value: {anova_p}")
    
    test_used = 'ANOVA'
    test_stat, test_p = anova_stat, anova_p
    
    # Post-hoc: Tukey HSD si hay diferencias significativas
    if anova_p < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Preparar datos para Tukey
        stacked_scores = bootstrap_matrix.flatten()
        group_labels = np.repeat(model_names, n_bootstrap)
        
        tukey_result = pairwise_tukeyhsd(stacked_scores, group_labels, alpha=0.05)
        print(f"\nPost-hoc Tukey HSD:")
        print(tukey_result)
        
        posthoc_result = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                    columns=tukey_result._results_table.data[0])
        
else:
    # Friedman si no todas las muestras son normales
    from scipy.stats import friedmanchisquare
    friedman_stat, friedman_p = friedmanchisquare(*[bootstrap_matrix[:, i] for i in range(len(model_names))])
    print(f"\nTest de Friedman en muestras bootstrap:")
    print(f"  Chi-square: {friedman_stat}")
    print(f"  p-value: {friedman_p}")
    
    test_used = 'Friedman'
    test_stat, test_p = friedman_stat, friedman_p
    
    # Post-hoc: Nemenyi si hay diferencias significativas
    if friedman_p < 0.05:
        import scikit_posthocs as sp
        
        # Usar la función correcta con los datos bootstrap
        nemenyi_result = sp.posthoc_nemenyi_friedman(bootstrap_matrix)
        
        print(f"\nPost-hoc Nemenyi:")
        print("Matriz de p-values:")
        nemenyi_result.index = model_names
        nemenyi_result.columns = model_names
        print(nemenyi_result.round(4))
        
        # Verificar que los p-values están en el rango correcto
        max_p = nemenyi_result.max().max()
        min_p = nemenyi_result.min().min()
        print(f"\nRango de p-values: [{min_p}, {max_p}]")
        
        if max_p > 1.0 or min_p < 0.0:
            print("⚠  ERROR: P-values fuera del rango válido [0, 1]")
        else:
            print("✅ P-values en el rango válido [0, 1]")
        
        posthoc_result = nemenyi_result
        
        # Interpretación de resultados
        alpha = 0.05
        print(f"\nComparaciones significativas (p < {alpha}):")
        significant_pairs = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                p_value = nemenyi_result.iloc[i, j]
                if p_value < alpha:
                    significant_pairs.append((model_names[i], model_names[j], p_value))
                    print(f"  {model_names[i]} vs {model_names[j]}: p = {p_value} *")
        
        if not significant_pairs:
            print("  No se encontraron diferencias significativas entre pares de modelos")

# Visualización de los resultados bootstrap
plt.figure(figsize=(5, 6))
plt.boxplot([bootstrap_matrix[:, i] for i in range(len(model_names))], 
           tick_labels=model_names)
plt.title('Distribución R² Bootstrap por Modelo')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("images/bootstrap_r2_dist.png")

# Subplot 2: Medias y errores estándar
plt.figure(figsize=(5, 6))
means = [np.mean(bootstrap_matrix[:, i]) for i in range(len(model_names))]
stds = [np.std(bootstrap_matrix[:, i]) for i in range(len(model_names))]
bars = plt.bar(model_names, means, yerr=stds, capsize=5, 
               color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
plt.title('Media R² ± Error Estándar (Bootstrap)')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
for bar, mean, std in zip(bars, means, stds):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01, 
             f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig("images/bootstrap_r2_media.png")

# Subplot 3: Heatmap de p-values si existe
if test_p < 0.05 and 'posthoc_result' in locals():
    plt.figure(figsize=(10, 6))
    if test_used == 'Friedman':
        sns.heatmap(posthoc_result, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   xticklabels=model_names, yticklabels=model_names)
        plt.title('Nemenyi - Matriz de p-values')
    else:
        # Para Tukey, crear una representación visual diferente
        plt.text(0.5, 0.5, 'Ver resultados Tukey\nen la consola', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Post-hoc Tukey HSD')
    plt.tight_layout()
    plt.savefig("images/bootstrap_posthoc.png")

# Subplot 4: Ranking visual
plt.figure(figsize=(10, 6))
ranking_scores = [np.mean(bootstrap_matrix[:, i]) for i in range(len(model_names))]
sorted_indices = np.argsort(ranking_scores)[::-1]
sorted_models = [model_names[i] for i in sorted_indices]
sorted_scores = [ranking_scores[i] for i in sorted_indices]

bars = plt.barh(range(len(sorted_models)), sorted_scores, 
               color=['gold', 'silver', '#CD7F32', 'lightgray'])
plt.yticks(range(len(sorted_models)), [f"{i+1}. {model}" for i, model in enumerate(sorted_models)])
plt.xlabel('R² Score Promedio')
plt.title('Ranking de Modelos')
plt.grid(axis='x', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/bootstrap_ranking.png')
plt.close()

print(f"\n📊 Análisis bootstrap completado")
print(f"\nResumen de resultados:")
print(f"  Test utilizado: {test_used}")
print(f"  Estadístico: {test_stat:.4f}")
print(f"  p-value: {test_p:.6f}")
if test_p < 0.05:
    print("  ✅ Diferencias significativas encontradas")
    print("  📋 Se realizó análisis post-hoc")
else:
    print("  ❌ No se encontraron diferencias significativas")
    print("  📋 No se requiere análisis post-hoc")

# MEJOR MODELO
best_model_name = comparison_df.iloc[0]['Modelo']
best_r2 = comparison_df.iloc[0]['R² Score']

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"R² Score: {best_r2:.4f}")
print(f"Esto significa que el modelo explica el {best_r2*100:.2f}% de la varianza en los datos.")

# GUARDAR MODELOS Y SCALERS
print("\n" + "="*60)
print("GUARDANDO MODELOS Y SCALERS")
print("="*60+"\n")

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

# Modelos usados
models_list = ['Regresión Lineal',
          'Bosques Aleatorios',
          'Regresión de Vectores de Soporte',
          'Potenciación de Gradiente']
models_list += ['XGBoost', 'LightGBM']
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

# Agregar coeficientes de regresión lineal al PDF
add_subtitle(story, "Coeficientes de la Regresión Lineal")
add_table(story, lr_coef)
add_spacer(story, 1, 6)

# Comparación de modelos
add_subtitle(story, "Comparación R2")
add_image(story, "images/comparacion_r2.png", 240, 300)
add_subtitle(story, "Comparación Error Cuadrado Promedio")
add_image(story, "images/comparacion_mse.png", 240, 300)
add_subtitle(story, "Comparación Precisión")
add_image(story, "images/comparacion_precision.png", 240, 300)
add_subtitle(story, "Comparación Brier Score")
add_image(story, "images/comparacion_brier.png", 240, 300)

# Análisis de clasificación
add_subtitle(story, "Análisis de clasificación")
add_paragraph(story, f"Métricas a utilizar:")
add_paragraph(story, f"• Coeficiente de Determinación (R2)")
add_paragraph(story, f"• Error Absoluto Promedio (MAE)")
add_paragraph(story, f"• Raíz del Error Absoluto Promedio (RMCE)")
add_paragraph(story, f"• Error Cuadrático Promedio (MCE)")
add_paragraph(story, f"• Puntuación de Brier (Brier)")
add_table(story, clasificacion_df)
add_spacer(story, 1,6)

# Pruebas estadísticas
add_subtitle(story, "Análisis Estadístico de Comparación de Modelos")
add_paragraph(story, "Se realizó un análisis estadístico robusto utilizando bootstrap para comparar el rendimiento de los modelos.")
add_spacer(story, 1, 6)

# Información del bootstrap
add_subtitle(story, "Metodología Bootstrap")
add_paragraph(story, f"Se generaron {n_bootstrap} muestras bootstrap para obtener distribuciones de rendimiento más robustas.")
add_paragraph(story, "Estadísticas bootstrap por modelo:")
for i, model in enumerate(model_names):
    mean_score = np.mean(bootstrap_matrix[:, i])
    std_score = np.std(bootstrap_matrix[:, i])
    add_paragraph(story, f"• {model}: Media R² = {mean_score:.4f} ± {std_score:.4f}")
add_spacer(story, 1, 6)

# Test de normalidad
add_subtitle(story, "Test de Normalidad en Muestras Bootstrap")
normality_data = []
for i, model in enumerate(model_names):
    if len(bootstrap_matrix[:, i]) <= 50:
        stat, p = shapiro(bootstrap_matrix[:, i])
        test = 'Shapiro-Wilk'
    else:
        stat, p = kstest(bootstrap_matrix[:, i], 'norm')
        test = 'Kolmogorov-Smirnov'
    
    normality_data.append({
        'Modelo': model,
        'Test': test,
        'Estadístico': f"{stat}",
        'p-value': f"{p}",
        'Normal': 'Sí' if p > 0.05 else 'No'
    })

normality_df = pd.DataFrame(normality_data)
add_table(story, normality_df)
add_paragraph(story, f"Todas las muestras bootstrap son normales: {'Sí' if all_bootstrap_normal else 'No'}")
add_spacer(story, 1, 6)

# Prueba estadística principal
add_subtitle(story, f"Comparación de Modelos: Test de {test_used}")
add_paragraph(story, f"Test utilizado: {test_used}")
add_paragraph(story, f"Estadístico: {test_stat}")
add_paragraph(story, f"p-value: {test_p}")

if test_p < 0.05:
    add_paragraph(story, "Resultado: Diferencias significativas encontradas entre los modelos (p < 0.05)")
    add_paragraph(story, "Se procede con el análisis post-hoc para identificar qué modelos difieren específicamente.")
else:
    add_paragraph(story, "Resultado: No se encontraron diferencias significativas entre los modelos (p ≥ 0.05)")
    add_paragraph(story, "No se requiere análisis post-hoc.")

add_spacer(story, 1, 6)

# Análisis post-hoc si es necesario
if test_p < 0.05 and 'posthoc_result' in locals():
    if test_used == 'ANOVA':
        add_subtitle(story, "Análisis Post-hoc: Tukey HSD")
        add_paragraph(story, "Se aplicó la prueba de Tukey HSD para comparaciones múltiples:")
        # Para Tukey, mostrar resultados textuales
        tukey_data = []
        for row in posthoc_result.itertuples():
            tukey_data.append({
                'Grupo 1': row[1],
                'Grupo 2': row[2],
                'Diferencia de medias': f"{row[3]:.4f}",
                'p-value': f"{row[6]:.4f}",
                'Significativo': 'Sí' if row[6] < 0.05 else 'No'
            })
        tukey_df = pd.DataFrame(tukey_data)
        add_table(story, tukey_df)
        
    else:  # Friedman
        add_subtitle(story, "Análisis Post-hoc: Nemenyi")
        add_paragraph(story, f"Dado que el test de Friedman fue significativo (p = {test_p:.2e}), se aplicó la prueba de Nemenyi.")
        
        # Crear tabla de p-values significativos
        alpha = 0.05
        significant_pairs = []
        pairwise_data = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                p_value = posthoc_result.iloc[i, j]
                pairwise_data.append({
                    'Modelo 1': model_names[i],
                    'Modelo 2': model_names[j],
                    'p-value': f"{p_value:.4f}",
                    'Significativo': 'Sí' if p_value < alpha else 'No'
                })
                if p_value < alpha:
                    significant_pairs.append((model_names[i], model_names[j], p_value))
        
        pairwise_df = pd.DataFrame(pairwise_data)
        add_table(story, pairwise_df)
        
        if significant_pairs:
            add_subtitle(story, "Comparaciones Significativas (p < 0.05)")
            for pair in significant_pairs:
                add_paragraph(story, f"• {pair[0]} vs {pair[1]}: p = {pair[2]:.4f}")
        else:
            add_paragraph(story, f"No se encontraron diferencias significativas entre pares de modelos individuales (α = {alpha})")
    
    add_spacer(story, 1, 6)

# Visualización del análisis bootstrap
add_subtitle(story, "Prueba de Nemenyi")
add_image(story, "images/bootstrap_posthoc.png", 500, 300)
add_spacer(story, 1, 6)

# Ranking final de modelos
add_subtitle(story, "Ranking Final de Modelos")
ranking_scores = [np.mean(bootstrap_matrix[:, i]) for i in range(len(model_names))]
sorted_indices = np.argsort(ranking_scores)[::-1]

ranking_data = []
for rank, idx in enumerate(sorted_indices):
    model = model_names[idx]
    score = ranking_scores[idx]
    std_score = np.std(bootstrap_matrix[:, idx])
    ranking_data.append({
        'Ranking': rank + 1,
        'Modelo': model,
        'R² Promedio': f"{score:.4f}",
        'Desviación Estándar': f"{std_score:.4f}",
        'Intervalo de Confianza': f"[{score-1.96*std_score:.4f}, {score+1.96*std_score:.4f}]"
    })

ranking_df = pd.DataFrame(ranking_data)
add_table(story, ranking_df)
add_spacer(story, 1, 6)

# Conclusiones estadísticas
add_subtitle(story, "Conclusiones del Análisis Estadístico")
add_paragraph(story, "Principales hallazgos del análisis estadístico:")

conclusions = [
    f"1. Se utilizó bootstrap con {n_bootstrap} muestras para obtener estimaciones robustas del rendimiento",
    f"2. El test de {test_used} {'confirmó' if test_p < 0.05 else 'no detectó'} diferencias significativas entre modelos (p = {test_p:.6f})",
]

if test_p < 0.05:
    conclusions.append("3. El análisis post-hoc identificó diferencias específicas entre pares de modelos")
    conclusions.append("4. El ranking final refleja diferencias estadísticamente significativas")
else:
    conclusions.append("3. Todos los modelos muestran rendimiento estadísticamente equivalente")
    conclusions.append("4. La selección del modelo puede basarse en otros criterios (interpretabilidad, velocidad, etc.)")

conclusions.append(f"5. El modelo '{model_names[sorted_indices[0]]}' obtuvo el mejor rendimiento promedio")

for conclusion in conclusions:
    add_paragraph(story, conclusion)

add_spacer(story, 1, 6)

# Mejor modelo
add_subtitle(story, "Modelo Optimo")
add_paragraph(story, f"<b>MEJOR MODELO:</b> {best_model_name}")
add_paragraph(story, f"• R2: {results[best_model_name]['r2']}")
add_paragraph(story, f"• Error Absoluto Promedio (MAE): {results[best_model_name]['mae']}")
add_paragraph(story, f"• Raiz del Error Cuadrado Promedio (RMCE): {results[best_model_name]['rmse']}")
add_paragraph(story, f"• Error Cuadrado Promedio (MCE): {results[best_model_name]['mse']}")
add_paragraph(story, f"• Brier Score: {results[best_model_name]['brier']}")

# Generar PDF
build_pdf(doc, story)   
print("\n✅ PDF generado correctamente")

def run_streamlit():
    os.system('streamlit run app.py --server.port 8501 --server.headless true -server.fileWatcherType none --browser.gatherUsageStats false')