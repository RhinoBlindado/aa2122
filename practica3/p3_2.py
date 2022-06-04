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

# FUNCIONES PARA EJERCICIOS

def encodeCycle(data, col, maxVal):
    data[col+"_sin"] = np.sin(2 * np.pi * data[col] / maxVal)
    data[col+"_cos"] = np.cos(2 * np.pi * data[col] / maxVal)
        
def monthToInt(data, col, monthDict):
    data[col] = data[col].map(monthDict)

def loadData(dataPath, sep=";", header=0):
    dataset = pd.read_csv(dataPath, sep=sep, header=header)
    return dataset

def processData(data, categorical, binary, cyclical, custom):
    
    if categorical is not None:
        data = pd.get_dummies(data, columns=categorical, drop_first=True)
    
    if binary is not None:
        for feat in binary:
            data[feat] = np.where(data[feat] == 'yes', 1, -1)
     
    if cyclical is not None:
        for feat in cyclical:
            encodeCycle(data, feat, cyclical[feat])
            data.drop(labels=[feat], axis=1, inplace=True)

    if custom is not None:
        for feat in custom:
            for key in feat:
                data[key] = data[key].map(feat.get(key))
        
    return data

def charsTagsSplit(data, tag):
    x = data.drop([tag], axis=1)
    y = data[tag]
    
    return x, y

def normalizeData(train, test, normCols, featRange=((-1, 1))):
    scaler = skpp.MinMaxScaler(feature_range=featRange)

    scaler.fit(train[normCols])
    train[normCols] = scaler.transform(train[normCols])
    test[normCols] = scaler.transform(test[normCols])


def printLabelFrec(df, column = None, testOverride = False, decimals=".2f"):
    
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
    plt.figure()
    
    scaleColor = lambda a, top : a / top 
    actColors = cm.rainbow(scaleColor(frecs, np.max(frecs)))
    
    plt.bar(labels, frecs, color=actColors)       
    plt.ylabel("Proporción en muestra")
    plt.xlabel("Etiquetas")
    plt.xticks(rotation=rotation)
 
    plt.title(title)
    plt.show()
    
    
def plotLearningCurve(sizes, train, test, title=None):
    plt.figure()
    
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

dataPath = "./datos/regresion/YearPredictionMSD.txt"

musicData = loadData(dataPath, ",", None)

#%% DIVISION DE DATOS
#####################

X, Y = charsTagsSplit(musicData, 0)

train_chars, test_chars, train_tags, test_tags = skms.train_test_split(X, Y, test_size=0.3)


#%% ANALISIS DE DATOS
#####################

print("Análisis de Datos:")
print("Mostrando diferentes estadísticas del conjunto de entrenamiento:")
with pd.option_context('display.max_columns', 8):
    print(train_chars.describe(include=np.number))
    print("\n")


print("Columna 'año'")
print(train_tags.describe())
labels, frecs = printLabelFrec(train_tags, testOverride=True, decimals=".4f")
plotLabelFrec(labels, frecs, title="Atributo 'año'")

#%% CODIFICAR LOS DATOS
#######################

pca = skd.PCA(n_components=0.95)

pca.fit(train_chars)

red_train_x = pca.transform(train_chars)
red_test_x  = pca.transform(test_chars)

train_x = train_chars.to_numpy()
train_y = train_tags.to_numpy()
test_x = test_chars.to_numpy()
test_y = test_tags.to_numpy()

#%% VALIDACION CRUZADA
######################

CVParamGrid =   [{
                    'loss' : ['squared_error'],
                    'alpha': [0.001, 0.01],
                    'eta0' : [0.01, 0.1],
                    'learning_rate' : ['adaptive','optimal'],
                }]


score = "r2"

LRGrid = skms.GridSearchCV(sklm.SGDRegressor(max_iter=10000), 
                       CVParamGrid, 
                       scoring=score,
                       refit="r2",
                       cv=3,
                       verbose=1,
                       n_jobs=-1)

# Esto se tarda un tiempo, aproximadamente 3 minutos.
print("Empezando la selección de mejor modelo con Cross Validation")
print("Tiempo estimado: 3 minutos\n")

LRGrid.fit(red_train_x, train_y)

print("Resultados de Cross-Validation:")
print("Mejor puntuación R^2: {}".format(LRGrid.best_score_))
print("Mejores parámetros: {}".format(LRGrid.best_params_))


#%% ENTRENAMIENTO
#################

bestLR = LRGrid.best_estimator_

bestLR.fit(red_train_x, train_y)
pred_train_y = bestLR.predict(red_train_x)

print("Conjunto de Entrenamiento:")
print("Error Cuadrático Medio: {:.4f}".format(skm.mean_squared_error(train_y, pred_train_y)))
print("Error R^2: {:.4f}".format(bestLR.score(red_train_x, train_y)))

#%% TEST
########

pred_test_y = bestLR.predict(red_test_x)

print("Conjunto de Test:")
print("Error Cuadrático Medio: {:.4f}".format(skm.mean_squared_error(test_y, pred_test_y)))
print("Error R^2: {:.4f}".format(bestLR.score(red_test_x, test_y)))


#%% CURVAS DE APRENDIZAJE
#########################

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