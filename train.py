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
from make_pdf import create_pdf, build_pdf, add_title, add_subtitle, add_paragraph, add_spacer, add_list, add_image, add_table
from scipy.stats import shapiro, kstest, f_oneway, friedmanchisquare, wilcoxon
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# Lectura de datos
ad_df = pd.read_csv("Datos_abiertos_admision_2021_1_2024_1.csv")

# Limpieza de columnas y eliminar filas con valores nulos
ad_df = ad_df.drop(['IDHASH', 'COLEGIO_DEPA', 'COLEGIO_PROV', 'COLEGIO_DIST',
                    'COLEGIO_PAIS', 'DOMICILIO_DEPA', 'DOMICILIO_PROV',
                    'DOMICILIO_DIST', 'NACIMIENTO_PAIS', 'NACIMIENTO_DEPA',
                    'NACIMIENTO_PROV', 'NACIMIENTO_DIST'], axis=1).dropna()

print("\n" + "="*60)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*60+"\n")
estadisticas = ad_df.describe()
print(estadisticas)

# Categorización de datos
columnas_categoricas = ['COLEGIO', 'ESPECIALIDAD', 'SEXO', 'MODALIDAD', 'INGRESO']
label_encoders = {}
for col in columnas_categoricas:
  le = LabelEncoder()
  col_name = col + "_ENCODED"
  ad_df[col_name] = le.fit_transform(ad_df[col])
  label_encoders[col] = le
  ad_df = ad_df.drop(col, axis=1)

# Visualización de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(ad_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.tight_layout()
plt.savefig("images/matriz_correlacion.png")

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
    'EAP': [results[model]['mae'] for model in results.keys()],
    'RECP': [results[model]['rmse'] for model in results.keys()],
    'ECP': [results[model]['mse'] for model in results.keys()],
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

# NORMALITY TESTS (Shapiro-Wilk and Kolmogorov-Smirnov)
print("\n" + "="*60)
print("TESTS DE NORMALIDAD DE RESIDUALES")
print("="*60)

normality_tests = []
for model_name in results.keys():
    residuals = y_test - results[model_name]['y_pred']
    if len(residuals) <= 50:
        stat, p = shapiro(residuals)
        test = 'Shapiro-Wilk'
    else:
        stat, p = kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        test = 'Kolmogorov-Smirnov'
        
    normality_tests.append({
        'model_name': model_name,
        'stat': stat,
        'p': p,
        'test': test
    })
    print(f"\nModelo: {model_name}")
    print(f"  {test}: stat={stat}, p={p}")

normality_df = pd.DataFrame(normality_tests)

# Determinar si los modelos pasan normalidad (p > 0.05)
all_normal = all(normality_df['p'] > 0.05)

# Prueba ANOVA o Friedman
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS: ANOVA o FRIEDMAN")
print("="*60)

residuals_list = [results[m]['y_pred'] - results[m]['y_real'] for m in results.keys()]
residuals_matrix = np.column_stack(residuals_list)
model_names = list(results.keys())

posthoc_result = None
posthoc_plot_path = None

if all_normal:
    # ANOVA
    anova_stat, anova_p = f_oneway(*residuals_list)
    print(f"\nANOVA: stat={anova_stat}, p={anova_p}")
    test_used = 'ANOVA'
    test_stat, test_p = anova_stat, anova_p
    # Post-hoc: Tukey HSD
    stacked_residuals = np.concatenate(residuals_list)
    group_labels = np.concatenate([[name]*len(residuals_list[0]) for name in model_names])
    tukey = pairwise_tukeyhsd(stacked_residuals, group_labels)
    posthoc_result = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    # Plot
    plt.figure(figsize=(15,9))
    sp.sign_plot(posthoc_result['reject'].astype(float).values.reshape(len(model_names),-1), model_names)
    plt.title('Tukey HSD Post-hoc')
    posthoc_plot_path = 'images/posthoc_tukey.png'
    plt.savefig(posthoc_plot_path)
    plt.close()
else:
    # Friedman test
    friedman_stat, friedman_p = friedmanchisquare(*[residuals_matrix[:,i] for i in range(residuals_matrix.shape[1])])
    print(f"\nFriedman: stat={friedman_stat}, p={friedman_p}")
    test_used = 'Friedman'
    test_stat, test_p = friedman_stat, friedman_p

    if friedman_p < 0.05:
        print("\n" + "="*60)
        print("ANÁLISIS POST-HOC: PRUEBA DE NEMENYI")
        print("="*60)
        
        # Aplicar prueba de Nemenyi
        nemenyi_result = sp.posthoc_nemenyi_friedman(residuals_matrix)
        posthoc_result = nemenyi_result
        
        # Mostrar matriz de p-values
        print("\nMatriz de p-values de Nemenyi:")
        print(nemenyi_result.round(4))
        
        # Interpretación de resultados
        print("\n" + "="*60)
        print("INTERPRETACIÓN DE RESULTADOS")
        print("="*60)
        
        alpha = 0.05
        significant_pairs = []
        
        print(f"\nComparaciones significativas (p < {alpha}):")
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                p_value = nemenyi_result.iloc[i, j]
                if p_value < alpha:
                    significant_pairs.append((model_names[i], model_names[j], p_value))
                    print(f"  {model_names[i]} vs {model_names[j]}: p = {p_value:.4f} *")
        
        if not significant_pairs:
            print("  No se encontraron diferencias significativas entre pares de modelos")
        
        # Ranking de modelos basado en rendimiento promedio
        print(f"\nRanking de modelos (basado en R² Score):")
        ranking_df = comparison_df.copy()
        ranking_df['Ranking'] = range(1, len(ranking_df) + 1)
        
        for idx, row in ranking_df.iterrows():
            print(f"  {row['Ranking']}. {row['Modelo']} (R² = {row['R² Score']:.4f})")
        
        # Grupos homogéneos (modelos sin diferencias significativas)
        print(f"\nGrupos homogéneos (sin diferencias significativas):")
        
        # Crear grupos basados en las comparaciones no significativas
        groups = []
        models_processed = set()
        
        for i, model1 in enumerate(model_names):
            if model1 not in models_processed:
                current_group = [model1]
                models_processed.add(model1)
                
                for j, model2 in enumerate(model_names):
                    if i != j and model2 not in models_processed:
                        p_value = nemenyi_result.iloc[i, j]
                        if p_value >= alpha:
                            current_group.append(model2)
                            models_processed.add(model2)
                
                if len(current_group) > 1:
                    groups.append(current_group)
        
        if groups:
            for i, group in enumerate(groups, 1):
                print(f"  Grupo {i}: {', '.join(group)}")
        else:
            print("  Todos los modelos son significativamente diferentes")
        
        # Crear gráfico mejorado
        plt.figure(figsize=(12, 8))
        
        # Heatmap de p-values
        plt.subplot(2, 1, 1)
        sns.heatmap(nemenyi_result, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                    xticklabels=model_names, yticklabels=model_names,
                    cbar_kws={'label': 'p-value'})
        plt.title('Prueba de Nemenyi - Matriz de p-values')
        
        # Gráfico de significancia
        plt.subplot(2, 1, 2)
        significance_matrix = (nemenyi_result < alpha).astype(int)
        sns.heatmap(significance_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
                    xticklabels=model_names, yticklabels=model_names,
                    cbar_kws={'label': 'Significativo (1) / No significativo (0)'})
        plt.title(f'Diferencias significativas (α = {alpha})')
        
        plt.tight_layout()
        posthoc_plot_path = 'images/posthoc_nemenyi_detailed.png'
        plt.savefig(posthoc_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de rangos promedio (Critical Difference plot)
        plt.figure(figsize=(10, 6))
        
        # Calcular rangos promedio para cada modelo
        avg_ranks = []
        for i in range(len(model_names)):
            # Simular ranking basado en R² (mejor modelo = menor rango)
            r2_scores = [results[model]['r2'] for model in model_names]
            sorted_indices = np.argsort(r2_scores)[::-1]  # Descendente
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
            avg_ranks.append(ranks[i])
        
        # Crear el gráfico
        y_pos = np.arange(len(model_names))
        bars = plt.barh(y_pos, avg_ranks, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        
        plt.yticks(y_pos, model_names)
        plt.xlabel('Rango Promedio (menor es mejor)')
        plt.title('Rangos Promedio de Modelos - Prueba de Nemenyi')
        plt.grid(axis='x', alpha=0.3)
        
        # Añadir valores en las barras
        for i, (bar, rank) in enumerate(zip(bars, avg_ranks)):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{rank:.1f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/nemenyi_ranks.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nGráficos guardados:")
        print(f"  - Matriz detallada: {posthoc_plot_path}")
        print(f"  - Rangos promedio: images/nemenyi_ranks.png")
        
    else:
        print("\nNo se realizó análisis post-hoc porque el test de Friedman no fue significativo")
        posthoc_result = None
        posthoc_plot_path = None
    """# Friedman
    friedman_stat, friedman_p = friedmanchisquare(*[residuals_matrix[:,i] for i in range(residuals_matrix.shape[1])])
    print(f"\nFriedman: stat={friedman_stat}, p={friedman_p}")
    test_used = 'Friedman'
    test_stat, test_p = friedman_stat, friedman_p
    # Post-hoc: Nemenyi
    nemenyi = sp.posthoc_nemenyi_friedman(residuals_matrix)
    posthoc_result = nemenyi
    # Plot
    plt.figure(figsize=(15,9))
    sp.sign_plot(nemenyi.values, model_names)
    plt.title('Nemenyi Post-hoc')
    posthoc_plot_path = 'images/posthoc_nemenyi.png'
    plt.savefig(posthoc_plot_path)
    plt.close()"""

# Residuales
plt.figure(figsize=(10,8))
plt.boxplot(residuals_list, tick_labels=model_names)
plt.title('Boxplot de residuales por modelo')
plt.ylabel('Residuales')
plt.tight_layout()
boxplot_path = 'images/boxplot_residuales.png'
plt.savefig(boxplot_path)
plt.close()

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
add_subtitle(story, "Comparación Precisión")
add_image(story, "images/comparacion_precision.png", 240, 300)
add_subtitle(story, "Comparación Brier Score")
add_image(story, "images/comparacion_brier.png", 240, 300)

# Análisis de clasificación
add_subtitle(story, "Análisis de clasificación")
add_table(story, clasificacion_df)
add_spacer(story, 1,6)

add_subtitle(story, "Test de normalidad de residuales (Shapiro-Wilk y Kolmogorov-Smirnov)")
normality_table = pd.DataFrame(normality_tests)
add_table(story, normality_table)
add_spacer(story, 1,6)

add_subtitle(story, "Comparación de modelos (ANOVA o Friedman)")
add_paragraph(story, f"Test usado: {test_used}")
add_paragraph(story, f"stat={test_stat}, p={test_p}")
if test_p < 0.05:
    add_paragraph(story, "\nDiferencias significativas entre modelos (p < 0.05)")
else:
    add_paragraph(story, "\nNo hay diferencias significativas entre modelos (p >= 0.05)")
add_spacer(story, 1,6)

if test_p < 0.05:
    add_subtitle(story, "Análisis Post-hoc: Prueba de Nemenyi")
    add_paragraph(story, f"Dado que el test de Friedman fue significativo (p = {test_p:.2e}), se procedió con el análisis post-hoc usando la prueba de Nemenyi.")
    add_spacer(story, 1,6)
    
    # Agregar matriz de p-values si existe
    if posthoc_result is not None:
        add_subtitle(story, "Matriz de p-values - Nemenyi")
        add_image(story, "images/posthoc_nemenyi_detailed.png", 400, 300)
        add_spacer(story, 1,6)
        
        add_subtitle(story, "Rangos promedio de modelos")
        add_image(story, "images/nemenyi_ranks.png", 400, 250)
        add_spacer(story, 1,6)
        
        # Interpretar resultados
        alpha = 0.05
        significant_pairs = []
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                p_value = posthoc_result.iloc[i, j]
                if p_value < alpha:
                    significant_pairs.append((model_names[i], model_names[j], p_value))
        
        if significant_pairs:
            add_subtitle(story, "Comparaciones significativas")
            for pair in significant_pairs:
                add_paragraph(story, f"\n\t• {pair[0]} vs {pair[1]}: p = {pair[2]:.2e}")
        else:
            add_paragraph(story, f"No se encontraron diferencias significativas entre pares de modelos (α = {alpha})")
        
        add_spacer(story, 1,6)
        
        add_subtitle(story, "Conclusiones del análisis post-hoc")
        add_paragraph(story, "La prueba de Nemenyi reveló las siguientes conclusiones:")
        
        add_paragraph(story, f"1. El test de Friedman confirmó diferencias significativas entre los modelos (p = {test_p:.2e})")
        add_paragraph(story, f"2. El modelo '{best_model_name}' mostró el mejor rendimiento con R² = {best_r2:.4f}")
        add_paragraph(story, "3. Los análisis post-hoc identificaron qué modelos difieren significativamente entre sí")
        add_paragraph(story, "4. Esta información es crucial para la selección del modelo óptimo")
else:
    add_paragraph(story, "No se realizó análisis post-hoc porque el test de Friedman no fue significativo.")

# Mejor modelo
add_subtitle(story, "Modelo Optimo")
add_paragraph(story, f"<b>MEJOR MODELO:</b> {best_model_name}")
add_paragraph(story, f"• R2: {results[best_model_name]['r2']}")
add_paragraph(story, f"• Error Absoluto Promedio: {results[best_model_name]['mae']}")
add_paragraph(story, f"• Raiz del Error Cuadrado Promedio: {results[best_model_name]['rmse']}")
add_paragraph(story, f"• Error Cuadrado Promedio: {results[best_model_name]['mse']}")
add_paragraph(story, f"• Brier Score: {results[best_model_name]['brier']}")

add_subtitle(story, "Gráfico de residuales por modelo")
add_image(story, boxplot_path, 400, 300)
add_spacer(story, 1,6)

# Generar PDF
build_pdf(doc, story)
print("\n✅ PDF generado correctamente")

def run_streamlit():
    os.system('streamlit run app.py --server.port 8501 --server.headless true -server.fileWatcherType none --browser.gatherUsageStats false')