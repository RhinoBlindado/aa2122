#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 3: Ajuste de Modelos Lineales - Regresión
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Mayo-Junio 2022
    
[ENGLISH]

    Practice 3: Fitting Linear Models - Regression
    Course: Machine Learning
    Author: Valentino Lugli (Github: @RhinoBlindado)
    May-June 2022
"""

# LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import sklearn.linear_model as sklm
import sklearn.preprocessing as skpp
import sklearn.model_selection as skms
import sklearn.metrics as skm
import sklearn.decomposition as skd
from tabulate import tabulate as tb

# Fijando la semilla
np.random.seed(1995)

## FUNCIONES AUXILIARES

def stop():
    """
    Detiene la ejecución hasta que se presione 'Enter'

    """
    input("Presiona Enter para continuar. . .")

def genBox(string):
    """
    Genera una caja de al rededor del texto para que se vea más estético.

    """
    strLen = len(string) + 2
    
    print("+",end='')
    for i in range(1, strLen - 1):
        print("-",end='')
    print("+")
    
    print("|{}|".format(string))
    
    print("+",end='')
    for i in range(1, strLen - 1):
        print("-",end='')
    print("+\n")

# FUNCIONES PARA EJERCICIOS

def encodeCycle(data, col, maxVal):
    """
    Codificar los datos en seno-coseno.

    Parameters
    ----------
    data : Pandas Dataframe
        Los datos a modificar
    col : String
        La columna a modificar.
    maxVal : Float/Int
        El valor máximo que puede tener la variable cíclica.

    Returns
    -------
    None.

    """
    data[col+"_sin"] = np.sin(2 * np.pi * data[col] / maxVal)
    data[col+"_cos"] = np.cos(2 * np.pi * data[col] / maxVal)
        
def monthToInt(data, col, monthDict):
    """
    Convertir los meses de cadena a entero.

    Parameters
    ----------
    data : Pandas Dataframe
        Los datos a modificar.
    col : String
        El nombre de la columna a modificar.
    monthDict : Diccionario.
        Diccionario que mapea las cadenas de los meses a su valor entero.

    Returns
    -------
    None.

    """
    data[col] = data[col].map(monthDict)

def loadData(dataPath, sep=";", header=0):
    """
    Cargar los datos a un Dataframe de Pandas.

    Parameters
    ----------
    dataPath : String
        Ruta hacia el fichero con los datos.
    sep : String, optional
        Separador de los datos, por defecto es ";".
    header : String, optional
        Como tratar la cabecera del fichero. Por defecto es 0.

    Returns
    -------
    dataset : TYPE
        DESCRIPTION.

    """
    dataset = pd.read_csv(dataPath, sep=sep, header=header)
    return dataset

def processData(data, categorical, binary, cyclical, custom):
    """
    Procesar los datos de un Dataframe

    Parameters
    ----------
    data : Pandas Dataframe
        Los datos originales.
    categorical : Lista.
        Lista con las variables categóricas para aplicarles One-Hot Encoding.
    binary : Lista.
        Lista con las variables binarias para cambiarlas de 'yes' y 'no' a -1, 1.
    cyclical : Lista de diccionario.
        Lista con las variables cíclicas para cambiarlas a codificación seno-coseno.
    custom : Lista de diccionario.
        Lista con variables y su mapeo particular.

    Returns
    -------
    data : Pandas Dataframe
        Datos modificados

    """
    
    # Si se han pasado variables categóricas...
    if categorical is not None:
        # ... utilizar la función de dummies apra el One-Hot Encoding.
        data = pd.get_dummies(data, columns=categorical, drop_first=True)
    
    # Si se han pasado binarias...
    if binary is not None:
        for feat in binary:
            # ... convertir de 'yes' y 'no' a '1' y '-1'.
            data[feat] = np.where(data[feat] == 'yes', 1, -1)
    
    # Ídem con cíclicas...
    if cyclical is not None:
        for feat in cyclical:
            # Codificar en seno-coseno.
            encodeCycle(data, feat, cyclical[feat])
            # Una vez se ha convertido, se quita la columna original.
            data.drop(labels=[feat], axis=1, inplace=True)
    
    # Ídem con mapeados personalizados...
    if custom is not None:
        for feat in custom:
            for key in feat:
                data[key] = data[key].map(feat.get(key))
        
    return data

def charsTagsSplit(data, tag):
    """
    Dividir los datos en las características X y las etiquetas Y.

    Parameters
    ----------
    data : Pandas Dataframe
        Los datos enteros.
    tag : String
        Nombre de la etiqueta del problema.

    Returns
    -------
    x : Pandas Dataframe
        Características.
    y : Pandas Dataframe
        Etiqueta.

    """
    x = data.drop([tag], axis=1)
    y = data[tag]
    
    return x, y

def normalizeData(train, test, normCols, featRange=((-1, 1))):
    """
    Normalizar los datos de entrenamiento y test, escalar los valores a valores
    entre -1 y 1.

    Parameters
    ----------
    train : Pandas Dataframe
        Características del conjunto de entrenamiento.
    test : Pandas Dataframe
        Características del conjunto de test.
    normCols : List
        Lista con los nombres de las columnas a escalar.
    featRange : Dupla, opcional.
        Dupla con los valorea a los que escalar los datos. Por defecto, ((-1, 1)).

    Returns
    -------
    None.

    """
    # Obtener el objeto que escala con los rangos.
    scaler = skpp.MinMaxScaler(feature_range=featRange)

    # Ajustar el objeto con los valores de entrenamiento.
    scaler.fit(train[normCols])
    
    # Con estos valores, escalar los datos de entrenamiento y test.
    train[normCols] = scaler.transform(train[normCols])
    test[normCols] = scaler.transform(test[normCols])


def printLabelFrec(df, column = None, testOverride = False, decimals=".2f"):
    """
    Función auxiliar para obtener las frecuencias de los distintos atributos
    de las variables categóricas.

    """
    if not testOverride:
        labels = np.unique(df[column])
        total = len(df[column])   
    else:        
        labels = np.unique(df)
        total = len(df)   

    frecs = []
    table = []
    
    for l in labels:
        if not testOverride:
            frec = len(np.where(df[column] == l)[0]) / total
        else:
            frec = len(np.where(df == l)[0]) / total

        frecs.append(frec)
        
        table.append([str(l), frec * 100])

    print(tb(table, headers=["Etiquetas", "%"], floatfmt=decimals))
    return labels, np.array(frecs)
    

def plotLabelFrec(labels, frecs, rotation="45", title=""):
    """
    Función auxiliar para mostrar en un histograma la frecuencia de unas 
    etiquetas categóricas

    """
    plt.figure()
    
    # Calcular el color dependiendo de la frecuencia del atributo.
    scaleColor = lambda a, top : a / top 
    actColors = cm.rainbow(scaleColor(frecs, np.max(frecs)))
    
    # Dibujarlo.
    plt.bar(labels, frecs, color=actColors)       
    plt.ylabel("Proporción en muestra")
    plt.xlabel("Etiquetas")
    plt.xticks(rotation=rotation)
 
    plt.title(title)
    plt.show()
    
    
def plotLearningCurve(sizes, train, test, title=None):
    """
    Función auxiliar para dibujar las curvas de aprendizaje

    """
    plt.figure()
    
    # Obtener los valores medios, que son los que serán dibujados.
    train = np.mean(train, axis=1)
    test  = np.mean(test, axis=1)
    
    plt.plot(sizes, train, 'o--', color='g', label="Entrenamiento")
    plt.plot(sizes, test, 'o-', color='r', label="Validación")
    
    plt.legend()

    plt.ylabel("Puntuación")
    plt.xlabel("Nº muestras")
    
    plt.title(title)
    plt.show()

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS


#%% CARGA DE DATOS
##################

genBox("Carga de datos")

dataPath = "./datos/regresion/YearPredictionMSD.txt"

musicData = loadData(dataPath, ",", None)

#%% DIVISION DE DATOS
#####################

genBox("División de Datos")

# En la columna 0 están las etiquetas.
X, Y = charsTagsSplit(musicData, 0)

# Realizar el split de entrenamiento y test.
train_chars, test_chars, train_tags, test_tags = skms.train_test_split(X, 
                                                                       Y, 
                                                                       test_size=0.3)


#%% ANALISIS DE DATOS
#####################

genBox("Análisis de datos")

print("Análisis de Datos:")
print("Mostrando diferentes estadísticas del conjunto de entrenamiento:")
with pd.option_context('display.max_columns', 8):
    print(train_chars.describe(include=np.number).round(decimals=2))
    print("\n")


print("Columna 'año'")
print(train_tags.describe().round(decimals=2))
labels, frecs = printLabelFrec(train_tags, testOverride=True, decimals=".4f")
plotLabelFrec(labels, frecs, title="Atributo 'año'")

#%% CODIFICAR LOS DATOS
#######################

# Como las variables van de 1 a 90, se saca un vector de [1, 2, ..., 90] 
# para escalar los valores al rango [-1, 1]
normalize = np.arange(1, 91)
normalizeData(train_chars, test_chars, normalize)

# Aplicar PCA y quedarse con el 95% de la varianza.
pca = skd.PCA(n_components=0.95)

# Usar los datos de entrenamiento para el fit.
pca.fit(train_chars)

# Y transformar los de entrenamiento y test por igual.
red_train_x = pca.transform(train_chars)
red_test_x  = pca.transform(test_chars)

train_x = train_chars.to_numpy()
train_y = train_tags.to_numpy()
test_x = test_chars.to_numpy()
test_y = test_tags.to_numpy()

#%% VALIDACION CRUZADA
######################

genBox("Validación Cruzada")

# Se definen los parámetros a explorar con regresión lineal.

CVParamGrid =   [{
                    'alpha': [0.001, 0.01],
                    'eta0' : [0.01, 0.1],
                    'learning_rate' : ['adaptive','optimal'],
                }]

# La forma que se desea valorar la calidad de la validación cruzada.
score = "r2"

# Definir el objeto, se indican más atributos que ya se fijan de una vez.

LRGrid = skms.GridSearchCV(sklm.SGDRegressor(loss="squared_error", 
                                             penalty="l2",
                                             max_iter=10000), 
                       CVParamGrid, 
                       scoring=score,
                       refit="r2",
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

# Esto se tarda un tiempo, aproximadamente 3 minutos.
print("Empezando la selección de mejor modelo con Cross Validation")
print("Tiempo estimado: 3 minutos\n")

# Ajustar los datos...
LRGrid.fit(red_train_x, train_y)

# ... obtener el modelo que tuvo la mejor puntuación de CV.
bestLR = LRGrid.best_estimator_

print("Resultados de Cross-Validation:")
print("Mejor puntuación R^2: {}".format(LRGrid.best_score_))
print("MSE: {}".format(skm.mean_squared_error(train_y, bestLR.predict(red_train_x))))
print("Mejores parámetros: {}".format(LRGrid.best_params_))


#%% ENTRENAMIENTO
#################

genBox("Entrenamiento")

bestLR.fit(red_train_x, train_y)
pred_train_y = bestLR.predict(red_train_x)

print("Conjunto de Entrenamiento:")
print("Error Cuadrático Medio: {:.4f}".format(skm.mean_squared_error(train_y, pred_train_y)))
print("Coeficiente R^2: {:.4f}".format(bestLR.score(red_train_x, train_y)))

#%% TEST
########

genBox("Test")

pred_test_y = bestLR.predict(red_test_x)

print("Conjunto de Test:")
print("Error Cuadrático Medio: {:.4f}".format(skm.mean_squared_error(test_y, pred_test_y)))
print("Coeficiente R^2: {:.4f}".format(bestLR.score(red_test_x, test_y)))


#%% CURVAS DE APRENDIZAJE
#########################

genBox("Curvas de Aprendizaje")

trainSizes, trainScore, testScore = skms.learning_curve(bestLR, 
                                                        red_train_x, 
                                                        train_y,
                                                        cv=3,
                                                        scoring="r2",
                                                        train_sizes=np.linspace(0.1, 1.0, 15),
                                                        n_jobs=-1)

plotLearningCurve(trainSizes, trainScore, testScore, title="$r^2$: Curva de aprendizaje")

trainSizes, trainScore, testScore = skms.learning_curve(bestLR, 
                                                        red_train_x, 
                                                        train_y,
                                                        cv=3,
                                                        scoring="neg_mean_squared_error",
                                                        train_sizes=np.linspace(0.1, 1.0, 15),
                                                        n_jobs=-1)

trainScore = trainScore * -1
testScore  = testScore  * -1
plotLearningCurve(trainSizes, trainScore, testScore, title="MSE: Curva de aprendizaje")