#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 3: Ajuste de Modelos Lineales - Clasificación
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Mayo-Junio 2022
    
[ENGLISH]

    Practice 3: Fitting Linear Models - Clasification
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

# Se definen diferentes listas con los atributos del dataset, identificando...

# - Atributos categóricos
categorical = ['job', 'marital', 'education', 'contact']

# - Atributos con un "mapeado" en perticular.
customMap = [{"poutcome" : { 'unknown' : 0, 'other' : 0, 'failure' : -1, 'success' : 1 }}]

# - Atributos que son binarios
binary = ['default', 'housing', 'loan']

# - Atributos cíclicos, indicando el valor máximo del ciclo. Asumiendo que todos
#   comienzan por 1.
cyclical = {'day' : 31, 'month' : 12}

#   + Mapeado de los meses a valores.
monthDict = {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5,
             'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10,
             'nov' : 11, 'dec' : 12}

# Ruta del fichero
dataPath = "./datos/clasificacion/bank-full.csv"

# Cargando los datos a un DataFrame de Pandas.
bankData = loadData(dataPath)

# Según la lectura de la documentación del dataset, se ha decidido eliminar
# esta columna. Más detalles en memoria.
bankData.drop(labels=["duration"], axis=1, inplace=True)


#%% DIVISION DE DATOS
#####################

# La columna a predecir es la 'y', se separa del resto de datos.
X, Y = charsTagsSplit(bankData, 'y')

# Se dividen los datos para ya tener el conjunto de entrenamiento y de test.
# - Al ser variables categóricas binarias, por seguridad se realiza una división
#   "stratified" para que se mantengan la proporción de las etiquetas de 'y'.
# - Se utiliza uno de los valores típicos de la división de los datos, 70-30.
train_chars, test_chars, train_tags, test_tags = skms.train_test_split(X, 
                                                                       Y, 
                                                                       stratify=Y,
                                                                       test_size=0.3)

#%% ANALISIS DE DATOS
#####################

print("Análisis de Datos:")
print("Mostrando diferentes estadísticas del conjunto de entrenamiento:")
with pd.option_context('display.max_columns', 32):
    print(train_chars.describe(include=np.number))
    print(train_chars.describe(exclude=np.number))
    print("\n")

print("Variables categóricas en más detalle:")
for i in categorical:
    print("Columna '{}'".format(i))
    labels, frecs = printLabelFrec(train_chars, i)
    plotLabelFrec(labels, frecs, title="Atributo {}".format(i))
    print("\n")
    
    
print("Variables binarias en más detalle:")
for i in binary:
    print("Columna '{}'".format(i))
    labels, frecs = printLabelFrec(train_chars, i)
    plotLabelFrec(labels, frecs, title="Atributo '{}'".format(i))
    print("\n")
    
    
print("Columna 'y'")
labels, frecs = printLabelFrec(train_tags, testOverride=True)
plotLabelFrec(labels, frecs, title="Atributo 'y'")


#%% CODIFICAR LOS DATOS
#######################

# Convertir los meses a sus valores enteros entre 1 y 12.
monthToInt(train_chars, "month", monthDict)
monthToInt(test_chars, "month", monthDict)

# Preprocesar los datos, las variables categoricas se cambian a one-hot encoding
# las binarias a -1 y 1, las cíclicas a codificación seno-coseno y las custom,
# dependiendo de lo definido en el map.
train_x = processData(train_chars, categorical, binary, cyclical, customMap)
test_x = processData(test_chars, categorical, binary, cyclical, customMap)

# Normalizar las variables numéricas, se realiza con MinMax de los datos de
# entrenamiento, se le aplica luego a los datos de test.
normalize = ["age", "balance", "pdays", "campaign", "previous"]
normalizeData(train_x, test_x, normalize)

# Se transforma de DataFrames de Pandas a arrays de Numpy.
train_x = train_x.to_numpy()
train_y = np.where(train_tags[:] == 'yes', 1, -1)
test_x = test_x.to_numpy()
test_y = np.where(test_tags[:] == 'yes', 1, -1)

#%% VALIDACION CRUZADA
######################


CVParamGrid =   [{
                    'loss' : ['log'],
                    'class_weight': [None, 'balanced'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'eta0' : [0.01, 0.1, 0.3],
                    'learning_rate' : ['adaptive','optimal'],
                }]


scores = ["accuracy", "precision", "recall", "f1"]

RLGrid = skms.GridSearchCV(sklm.SGDClassifier(), 
                       CVParamGrid, 
                       scoring=scores,
                       refit="f1",
                       cv=10,
                       verbose=1,
                       n_jobs=-1)

RLGrid.fit(train_x, train_y)

print("Resultados de Cross-Validation:")
print("Mejor puntuación F1: {}".format(RLGrid.best_score_))
print("Mejores parámetros: {}".format(RLGrid.best_params_))

#%% ENTRENAMIENTO
#################

bestRL = RLGrid.best_estimator_

bestRL.fit(train_x, train_y)
pred_train_y = bestRL.predict(train_x)

print("Conjunto de Entrenamiento:")
print("Error de Entropía cruzada: {:.4f}".format(skm.log_loss(train_y, pred_train_y)))
print("Accuracy: {:.4f}".format(bestRL.score(train_x, train_y)))
print("F1: {:.4f}".format(skm.f1_score(train_y, pred_train_y)))

#%% TEST
########

pred_test_y = bestRL.predict(test_x)

print("Conjunto de Test:")
print("Error de Entropía cruzada: {:.4f}".format(skm.log_loss(test_y, pred_test_y)))
print("Accuracy: {:.4f}".format(bestRL.score(test_x, test_y)))
print("F1: {:.4f}".format(skm.f1_score(test_y, pred_test_y)))


#%% ANALISIS DE RESULTADOS
##########################

print("Matriz de Confusión: \n")
print(skm.confusion_matrix(test_y, pred_test_y), "\n")

print("Reporte de clasificación:")
print(skm.classification_report(test_y, pred_test_y))


#%% CURVAS DE APRENDIZAJE
#########################

trainSizes, trainScore, testScore = skms.learning_curve(bestRL, 
                                                        train_x, 
                                                        train_y,
                                                        cv=10,
                                                        scoring="f1",
                                                        train_sizes=np.linspace(0.1, 1.0, 15),
                                                        n_jobs=-1)

plotLearningCurve(trainSizes, trainScore, testScore, title="F1: Curva de aprendizaje")

trainSizes, trainScore, testScore = skms.learning_curve(bestRL, 
                                                        train_x, 
                                                        train_y,
                                                        cv=10,
                                                        scoring="accuracy",
                                                        train_sizes=np.linspace(0.1, 1.0, 15),
                                                        n_jobs=-1)

plotLearningCurve(trainSizes, trainScore, testScore, title="Accuracy: Curva de aprendizaje")
