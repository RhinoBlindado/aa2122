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

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

def loadData(dataPath):
    dataset = pd.read_csv(dataPath, sep=";", header=0)
    dataset = pd.get_dummies
    

#%% EJERCICIO 1
#############

categorical = ['job', 'marital', 'education', 'contact', 'poutcome']
binary = ['default', 'housing', 'loan', 'y']

dataPath = "./datos/clasificacion/bank.csv"

# xFeats, yTags = loadData(dataPath)

dataset = pd.read_csv(dataPath, sep=";", header=0)
dataset = pd.get_dummies(dataset, columns=categorical, drop_first=True)

for feat in binary:
    dataset[feat] = np.where(dataset[feat] == 'yes', 1, -1)


