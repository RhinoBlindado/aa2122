#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 3: Ajuste de Modelos Lineales - Clasificación
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Mayo 2022
    
[ENGLISH]

    Practice 3: Fitting Linear Models - Clasification
    Course: Machine Learning
    Author: Valentino Lugli (Github: @RhinoBlindado)
    May 2022
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

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

def encodeCycle(data, col, maxVal):
    data[col+"_sin"] = np.sin(2 * np.pi * data[col] / maxVal)
    data[col+"_cos"] = np.cos(2 * np.pi * data[col] / maxVal)
        
def monthToInt(data, col, monthDict):
    data[col] = data[col].map(monthDict)

def loadData(dataPath):
    dataset = pd.read_csv(dataPath, sep=";", header=0)
    return dataset

def processData(data, categorical, binary, cyclical, custom):
    data = pd.get_dummies(data, columns=categorical, drop_first=True)

    for feat in binary:
        data[feat] = np.where(data[feat] == 'yes', 1, -1)
        
    for feat in cyclical:
        encodeCycle(data, feat, cyclical[feat])
        data.drop(labels=[feat], axis=1, inplace=True)

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


def plotRoc(tpr, fpr, thr):
    plt.figure()
    plt.plot(tpr)

#%% CARGA DE DATOS
##################

dataPath = "./datos/clasificacion/bank-full.csv"

bankData = loadData(dataPath)


#%% ANALISIS DE LOS DATOS
#########################





#%% CODIFICAR LOS DATOS
#######################

categorical = ['job', 'marital', 'education', 'contact']
customMap = [{"poutcome" : { 'unknown' : 0, 'other' : 0, 'failure' : -1, 'success' : 1 }}]
binary = ['default', 'housing', 'loan', 'y']
cyclical = {'day' : 31, 'month' : 12}

monthDict = {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5,
             'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10,
             'nov' : 11, 'dec' : 12}

monthToInt(bankData, "month", monthDict)

bankData.drop(labels=["duration"], axis=1, inplace=True)
problemData = processData(bankData, categorical, binary, cyclical, customMap)

#%% PREPROCESADO DE DATOS
#########################

X, Y = charsTagsSplit(problemData, 'y')

train_chars, test_chars, train_tags, test_tags = skms.train_test_split(X, 
                                                                       Y, 
                                                                       test_size=0.3,
                                                                       stratify=True,
                                                                       random_state=16)

normalize = ["age", "balance", "pdays", "campaign", "previous"]
normalizeData(train_chars, test_chars, normalize)

train_x = train_chars.to_numpy()
train_y = train_tags.to_numpy()
test_x = test_chars.to_numpy()
test_y = test_tags.to_numpy()

#%% VALIDACION CRUZADA
######################

CVParamGrid =   [{
                    'loss' : ['log'],
                    'class_weight': [None, 'balanced'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'eta0' : [0.01, 0.1, 0.3],
                    'learning_rate' : ['adaptive','optimal'],
                }]


scores = ["accuracy", "precision", "recall", "average_precision"]

RLGrid = skms.GridSearchCV(sklm.SGDClassifier(), 
                       CVParamGrid, 
                       scoring=scores,
                       refit="average_precision",
                       cv=10,
                       verbose=1,
                       n_jobs=-1)

RLGrid.fit(train_x, train_y)



#%% ENTRENAMIENTO
#################
bestRL = RLGrid.best_estimator_

bestRL.fit(train_x, train_y)

#%% TEST
########


#%% ANALISIS DE RESULTADOS
##########################
pred_test_y = bestRL.predict(test_x)

print(skm.classification_report(test_y, pred_test_y))
    