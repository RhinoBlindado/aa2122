#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 3: Ajuste de Modelos Lineales - Regresión
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Mayo 2022
    
[ENGLISH]

    Practice 3: Fitting Linear Models - Regression
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
import sklearn.model_selection as skml
import sklearn.metrics as skm

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

def encodeCycle(data, col, maxVal):
    data[col+"_sin"] = np.sin(2 * np.pi * data[col] / maxVal)
    data[col+"_cos"] = np.cos(2 * np.pi * data[col] / maxVal)
        
def monthToInt(data, col, monthDict):
    data[col] = data[col].map(monthDict)

def loadData(dataPath, sepIn=";", headerIn=None):
    dataset = pd.read_csv(dataPath, sep=sepIn, header=headerIn)
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

dataPath = "./datos/regresion/YearPredictionMSD.txt"

musicData = loadData(dataPath, ",")

#%% CODIFICAR LOS DATOS
#######################

#%%

X, Y = charsTagsSplit(musicData, 0)

train_chars, test_chars, train_tags, test_tags = skml.train_test_split(X, Y, test_size=0.3)

normalize = np.arange(1, 91)
normalizeData(train_chars, test_chars, normalize)

#%%

train_x = train_chars.to_numpy()
train_y = train_tags.to_numpy()
test_x = test_chars.to_numpy()
test_y = test_tags.to_numpy()