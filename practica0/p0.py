#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Practica 0: Introduccion a Python, Matplotlib y NumPy
    Asignatura: Aprendizaje Automatico
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Febrero-Marzo 2022
    
[ENGLISH]
    Practice 0: Introduction to Python, Matplotlib and NumPy
    Course: Machine Learning
    Author: Valentino Lugli (Github: @RhinoBlindado)
    February-March 2022
"""


# LIBRERIAS

# - Importando Numpy
import numpy as np

# - Importando scikit-learn general
import sklearn as sk

# - Importanto el dataset de Iris
from sklearn.datasets import load_iris

# - Importando Matplotlib
import matplotlib.pyplot as plt

# FUNCIONES AUXILIARES

# FIN FUNCIONES AUXILIARES

# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 2

def trainTestSplit(x_input, y_input, split):
    pass    
    
    # return x_train, y_train, x_test, y_test

## FUNCIONES EJER 2 FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1
###############

# 1.1: Leer la base de datos de Iris que hay en scikit-learn.

# - Leyendo el dataset de iris, esto da un objeto "Bunch" de scikit que 
#   contiene todo lo que se necesita del dataset.

iris = load_iris()

# 1.2: Obtener las características (datos de entrada X) y la clase (y).

# - Obteniendo las características de entrada, eso es el atributo 'data'
#   del Bunch.
# - Ídem para las clases del dataset, en este caso 'target'.

x_feat = iris.data
y_labels = iris.target

# 1.3: Quedarse con las características primera y tercera

# - Utilizando indexado, se obtienen de todos los datos de entrada [:, la
#   primera y tercera característica, es decir el indice 0 y 2, por lo tanto
#   se recorre la lista entera de dos en dos, comenzando en 0, ::2].

x_featReduced = x_feat[:,::2]

# 1.4: Visualizar con un Scatter Plot los datos, coloreando cada clase con un 
#      color diferente e indicando con una leyenda la clase a la que 
#      corresponde cada color. 

# - Creando una figura
plt.figure()

# - Realizando el scatter plot con los datos indicados, 'x' es la primera 
#   característica y 'y' la tercera.
# - 'c' indica la clase a la que pertenece cada dupla (x,y).
# - cmap indica como colorear las distintas clases.

scatter = plt.scatter(x_featReduced[:, 0], x_featReduced[:, 1], c=y_labels,
                       cmap='Set1')

# Renombrando los ejes de la figura acorde a los datos.
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])

# - Añadiendo las legendas correspondientes, se obtienen los datos del scatter
#   plot sobre las clases y se añade con 'labels' los nombres correspondientes.
# - Se utiliza '.tolist()' porque las arrays de Numpy generan un error con
#   Matplotlib.
plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names.tolist())
plt.show()


#%% EJERCICIO 2
###############

# 2.1: Separar en training (80 % de los datos) y test (20 %) aleatoriamente, 
# conservando la proporción de elementos en cada clase tanto en training 
# como en test.



#%% EJERCICIO 3
###############

#%% EJERCICIO 4
###############