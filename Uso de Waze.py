"""
El conjunto de datos ofrece una visión completa de las interacciones de los usuarios dentro de la aplicación de navegación Waze, crucial 
para comprender y mitigar la pérdida de usuarios. Waze, reconocido por sus servicios de navegación gratuitos, fomenta una comunidad 
dinámica de contribuyentes, incluidos editores de mapas, probadores beta y socios, unidos en la misión de mejorar la eficiencia y 
seguridad de los viajes globales.

Ideal tanto para el análisis de datos exploratorios (EDA) como para el aprendizaje automático (ML) , el conjunto de datos permite el 
desarrollo de modelos precisos para identificar los factores que contribuyen a la deserción. Estos modelos abordan preguntas críticas 
como quién, por qué y cuándo los usuarios abandonan , potenciando estrategias de retención proactivas."""

import os

#En este cuaderno, realizaré un análisis de datos exploratorio (EDA) en el conjunto de datos de Wase para obtener información valiosa para 
# futuras estadísticas.
# Análisis de regresión y aprendizaje automático.

#Importaciones y carga de datos
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('waze_app_dataset.csv')

#Exploración y limpieza de datos
print(df.head(10))

print(df.shape)

print(df.describe())

print(df.info())

df.isna().sum().sort_values(ascending=False)

df['label'].value_counts(normalize=True, dropna=False)

"""Visualizaciones
Los diagramas de caja serán útiles para determinar los valores atípicos y dónde reside la mayor parte de los puntos de datos en términos 
de unidades, sesiones y todos los demás.
variables numéricas continuas

Los histogramas son esenciales para comprender la distribución de variables.

Los diagramas de dispersión serán útiles para visualizar las relaciones entre variables.

Los gráficos de barras son útiles para comunicar niveles y cantidades, especialmente para información categórica.

sesiones"""

# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['sessions'], fliersize=1)
plt.title('Diagrama de caja de sesiones\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
plt.figure(figsize=(15,15))
sns.histplot(x=df['sessions'], color = "plum", edgecolor = "violet")
median = df['sessions'].median()
plt.axvline(median, color='red', linestyle='--')
plt.text(75,1200, 'median=56.0', color='red')
plt.title('Histograma de sesiones\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['drives'], fliersize=1)
plt.title('Impulsa el diagrama de caja\n', fontsize = '16', fontweight = 'bold')
plt.show()

def histogrammer(column_str, median_text=True, **kwargs):    
                                                             
    median=round(df[column_str].median(), 1)
    plt.figure(figsize=(15,15))
    ax = sns.histplot(x=df[column_str], **kwargs)          
    plt.axvline(median, color='red', linestyle='--')         
    if median_text==True:                                  
        ax.text(0.25, 0.85, f'median={median}', color='red',
            ha='left', va='top', transform=ax.transAxes)
    else:
        print('Mediana:', median)
    plt.title(f'{column_str} histograma\n', fontsize = '16', fontweight = 'bold')
histogrammer('drives')
plt.show()

# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['total_sessions'], fliersize=1)
plt.title('Diagrama de caja de sesiones totales\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
histogrammer('total_sessions')

# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=1)
plt.title('Diagrama de caja de n_days_after_onboarding\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
histogrammer('n_days_after_onboarding', median_text=False)

#driven_km_drives
# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['driven_km_drives'], fliersize=1)
plt.title('Diagrama de caja de drives_km_drives\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
histogrammer('driven_km_drives')

#duración_minutos_viajes
# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['duration_minutes_drives'], fliersize=1)
plt.title('Diagrama de caja de duración_minutos_unidades\n', fontsize = '16', fontweight = 'bold')
plt.show()

histogrammer('duration_minutes_drives')

#días_actividad
# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['activity_days'], fliersize=1)
plt.title('Diagrama de caja de días_actividad\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
histogrammer('activity_days', median_text=False, discrete=True)

#días_de_conducción
# Diagrama de caja
plt.figure(figsize=(15,15))
sns.boxplot(x=df['driving_days'], fliersize=1)
plt.title('Diagrama de caja de días_de_conducción\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Histograma
histogrammer('driving_days', median_text=False, discrete=True)

#dispositivo
#Esta es una variable categórica, por lo que no trazamos un diagrama de caja para ella. Un buen gráfico para una variable categórica binaria 
# es un gráfico circular.

# Gráfico circular
fig = plt.figure(figsize=(15,15))
data=df['device'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Usuarios por dispositivo\n', fontsize = '16', fontweight = 'bold')
plt.show()

#etiqueta
# Gráfico circular
fig = plt.figure(figsize=(15,15))
data=df['label'].value_counts()
plt.pie(data,
        labels=[f'{data.index[0]}: {data.values[0]}',
                f'{data.index[1]}: {data.values[1]}'],
        autopct='%1.1f%%'
        )
plt.title('Recuento de retenidos versus abandonos\n', fontsize = '16', fontweight = 'bold')
plt.show()

#driving_days vs. activity_days
#Debido a que tanto los días_de_conducción como los días_de_actividad representan recuentos de días durante un mes y también están 
# estrechamente relacionados, podemos trazarlos juntos en un solo histograma. Esto ayudará a comprender mejor cómo se relacionan entre 
# sí sin tener que desplazarse hacia adelante y hacia atrás comparando histogramas en dos lugares diferentes.

# Histograma
plt.figure(figsize=(15,15))
label=['driving days', 'activity days']
plt.hist([df['driving_days'], df['activity_days']],
         bins=range(0,33),
         label=label)
plt.xlabel('días\n')
plt.ylabel('contar\n')
plt.legend()
plt.title('días_de_conducción vs. días_de_actividad\n', fontsize = '16', fontweight = 'bold')
plt.show()

print(df['driving_days'].max())
print(df['activity_days'].max())

# Gráfico de dispersión
sns.scatterplot(data=df, x='driving_days', y='activity_days')
plt.title('días_de_conducción vs. días_de_actividad\n', fontsize = '16', fontweight = 'bold')
plt.plot([0,31], [0,31], color='red', linestyle='--')
plt.show()

#Retención por dispositivo
# Histograma
plt.figure(figsize=(15,15))
sns.histplot(data=df,
             x='device',
             hue='label',
             multiple='dodge',
             shrink=0.9
             )
plt.title('Retención por histograma del dispositivo\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Retención por kilómetros recorridos por día de conducción
# 1. Crear la columna `km_per_driving_day`
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Llamar a `describe()` en la nueva columna
df['km_per_driving_day'].describe()

# 1. Convertir valores infinitos a cero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0

# 2. Confirmamos que funcionó
df['km_per_driving_day'].describe().round(2)

# Histograma
plt.figure(figsize=(15,15))
sns.histplot(data=df,
             x='km_per_driving_day',
             bins=range(0,1201,20),
             hue='label',
             multiple='fill')
plt.ylabel('%', rotation=0)
plt.title('Tasa de abandono en kilómetros medios por día de conducción\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Tasa de abandono por número de días de conducción
# Histograma
plt.figure(figsize=(15,15))
sns.histplot(data=df,
             x='driving_days',
             bins=range(1,32),
             hue='label',
             multiple='fill',
             discrete=True)
plt.ylabel('%', rotation=0)
plt.title('Tasa de abandono por día de conducción\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Proporción de sesiones que ocurrieron en el último mes
#Creamos una nueva columna percent_sessions_in_last_month que representa el porcentaje del total de sesiones de cada usuario que iniciaron 
# sesión su último mes de uso.

df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# Valor madiano
df['percent_sessions_in_last_month'].median()

# Histograma
histogrammer('percent_sessions_in_last_month',
             hue=df['label'],
             multiple='layer',
             median_text=False)

df['n_days_after_onboarding'].median()

# Histograma
data = df.loc[df['percent_sessions_in_last_month']>=0.4]
plt.figure(figsize=(15,15))
sns.histplot(x=data['n_days_after_onboarding'])
plt.title('Núm. días después de la incorporación para usuarios con >=40% de sesiones en el último mes\n', fontsize = '16', fontweight = 'bold')
plt.show()

#Manejo de valores atípicos
#​Como se observó anteriormente, varias variables presentan valores atípicos. Para solucionar este problema, podemos crear una función que calcule el número 95.
# percentil de una columna especificada. Posteriormente, podemos sustituir los valores superiores al percentil 95 por el valor del percentil 95.
# percentil. Este enfoque garantiza que los valores extremos se sustituyan por un valor representativo de la distribución, es decir, el
# percentil 95.​

def outlier_imputer(column_name, percentile):
    # Cálculo del umbral
    threshold = df[column_name].quantile(percentile)
    # Imputar el umbral para valores > que el umbral
    df.loc[df[column_name] > threshold, column_name] = threshold

    print('{:>25} | percentil: {} | límite: {}'.format(column_name, percentile, threshold))
for column in ['sessions', 'drives', 'total_sessions',
               'driven_km_drives', 'duration_minutes_drives']:
               outlier_imputer(column, 0.95)
               
print(df.describe())

#encontrar valores nulos
df.isna().sum()

df['label'].value_counts()

df['device'].value_counts()

#Codificación de datos categóricos
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['device'] = le.fit_transform(df['device'])
df['label'] = le.fit_transform(df['label'])
df.head()

#trazar matriz de correlación
correlation_matrix = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidths=.5, ax=ax)
plt.show()

#dropping actividad_días ya que tiene una alta correlación con los días de conducción (casi el 95%)
df.drop('activity_days', axis=1, inplace=True)
print(df)

#trazar valores atípicos

for column in df:

    plt.figure()
    df.boxplot(column=column, grid=True)


    plt.title(f'Diagrama de caja para {column}\n', fontsize = '16', fontweight = 'bold')
    plt.xlabel('Valor')
    plt.ylabel(column)


    plt.show()
    
#eliminar valores atípicos

def remove_outliers(df, columns):
    for column in columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR


        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df


df = remove_outliers(df, df.columns)

"""Conclusión
​El análisis reveló que la tasa de abandono general es ~17%, y que esta tasa es consistente entre los usuarios de iPhone y los usuarios de 
Android.​Además, EDA ha revelado que los usuarios que conducen distancias muy largas durante sus días de conducción tienen más 
probabilidades de abandonar, pero los usuarios que conducen con más frecuencia son más propensos a abandonar el servicio.
menos probabilidades de abandonar. La razón de esta discrepancia es una oportunidad para una mayor investigación, y sería algo más que 
preguntar.
el equipo de datos de Waze sobre."""