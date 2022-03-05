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

# - Importando para hacer uso de funciones matemáticas más complejas.
import math

# FUNCIONES AUXILIARES

# FIN FUNCIONES AUXILIARES

# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 2

def trainTestSplit(x_input, y_input, split, className = [None]):
    
    # - Obtener las clases del vector de salida, son los números únicos
    # que están, se obtienen con la función 'unique'.
    classes = np.unique(y_input)
        
    # - Se obtiene el tamaño final de los vectores de test y train,
    # en proporción.
    # - Para que no haya problemas con el redondeo, se utiliza floor.
    # y el restante para la otra clase.
    
    numTrain = math.floor(len(x_input) * split)
    numTest = len(x_input) - numTrain
    
    # Se generan los arrays que contendrán los datos divididos.
    x_train = np.zeros( (numTrain, x_input.shape[1]) )
    x_test =  np.zeros( (numTest, x_input.shape[1]) )
    
    # Ídem para el vector de las etiquetas.
    y_train = np.zeros( (numTrain,) )
    y_test = np.zeros( (numTest,) )
    
    # Por cada clase...
    for i in classes:
        # Se obtienen los datos de entrada que corresponden a la clase
        # i-ésima.
        tempClass = x_input[ np.where(y_labels == i) ]
        
        # Se realiza un barajeo de los datos para desordenarlos dentro de su
        # propia clase.
        np.random.shuffle(tempClass)
        
        # Se obtiene la proporción de datos que irán al conjunto de train y
        # test dentro de la clase.
        trainDiv = math.floor(len(tempClass) * split)
        testDiv = len(tempClass) - trainDiv
        
        # Se calcula el desplazamiento para añadir los datos a los arrays
        # divididos, en este momento estarán en orden por cada clase.
        
        # Después que termine el bucle se volverán a barajear.
        
        startTrain = i * trainDiv
        endTrain = startTrain + trainDiv
        
        startTest = i * testDiv
        endTest = startTest + testDiv

        # Como los datos han sido ya barajeados in-place, se puede obtener
        # los primeros datos hasta la proporción requerida y se añaden al
        # conjunto de entrenamiento, y el resto al conjunto de test comodamente
        # pues ya no están en orden.
        
        x_train[startTrain:endTrain] = tempClass[:trainDiv]
        x_test[startTest:endTest] = tempClass[trainDiv:]
        
        # Como el bucle es por cada clase, a cada dato le corresponderá la 
        # clase i-ésima.
        
        y_train[startTrain:endTrain] = i
        y_test[startTest:endTest] = i
        
        if className[0] != None:
            
            print("--- Clase",className[i],"---")
            print("Ejemplos train:", len(tempClass[:trainDiv]))
            print("Ejemplos test:", len(tempClass[trainDiv:]))
        
    # Se obtienen índices con una permutación.
    permutTrain = np.random.permutation(numTrain)
    permutTest = np.random.permutation(numTest)
    
    # Se desordenan nuevamente los elementos pero sin perder la correspondencia
    # entre x e y.
    
    x_train = x_train[permutTrain]
    y_train = y_train[permutTrain]
    
    x_test = x_test[permutTest]
    y_test = y_test[permutTest]
    
    return x_train, y_train, x_test, y_test

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

iris = load_iris()

x_feat = iris.data
y_labels = iris.target

x_train, y_train, x_test, y_test = trainTestSplit(x_feat, y_labels, 0.8, 
                                                  iris.target_names)

print("Clase de los ejemplos de entrenamiento:", y_train)
print("Clase de los ejemplos de test:", y_test)

#%% EJERCICIO 3
###############

# 3.1: Obtener 100 valores equiespaciados entre 0 y 4pi
values = np.linspace(0, math.pi * 4, 100)

# 3.2: Obtener el valor de 10^−5 · sinh(x), cos(x) y 
# tanh(2 · sin(x) − 4 · cos(x)) para los 100 valores anteriormente calculados.

hyperSine = math.pow(10, -5) * np.sinh(values)
cosine = np.cos(values)
hyperTanSinCos = np.tanh(2 * np.sin(values) - 4 * np.cos(values))

# 3.3: Visualizar las tres curvas simultáneamente en el mismo plot

plt.figure()
plot = plt.plot(values, hyperSine, 'g--', values, cosine, 'k--', values, hyperTanSinCos, 'r--')
plt.legend(labels=["y = 1e-5 * sinh(x)", "y = cos(x)", "y = tanh(2sin(x) - 4cos(x)"], fontsize=8)
plt.show()

#%% EJERCICIO 4
###############

# 4.1: Funciones

# - f(x, y) = 1 - |x + y| - |y - x|
x1 = np.arange(-6, 6, 0.25)
y1 = np.arange(-6, 6, 0.25)

x1, y1 = np.meshgrid(x1, y1)

z1 = 1 - np.abs(x1 + y1) - np.abs(y1 - x1)
    
# - f(x, y) = x * y * e^(-x^2-y^2)
x2 = np.arange(-2, 2, 0.0625)
y2 = np.arange(-2, 2, 0.0625)

x2, y2, = np.meshgrid(x2, y2)

z2 = x2 * y2 * np.exp(-np.power(x2, 2)-np.power(y2,2))

# 4.2: Dibujar en 3D las funciones calculadas.

# - Generar la figura, se añade el tamaño en pulgadas y su DPI.
fig = plt.figure(figsize=(15, 10), dpi=300)

# - Como son dos figuras, se añade dos subplots, se indica con el 2. 
# - Como es la  primera figura, se utiliza el 1 al final.
ax = fig.add_subplot(1, 2, 1, projection='3d')

# - Se dibuja la figura con los elementos calculados.
surf = ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

# - Se delimita la figura por sus lados.
ax.set_zlim(-10, 0)
ax.set_xlim(-7, 7)
ax.set_ylim(-7, 7)

# - Añadiendo el título.
plt.title("Pirámide")

# - Se añade el segundo subplot, se indica ahora con el 2.
ax = fig.add_subplot(1, 2, 2, projection='3d')

# - Se dibuja, se cambia el mapa de color a viridis.
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)

# - Se delimitan los ejes.
ax.set_zlim(-0.15, 0.15)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# - Titulo
plt.title("$x \cdot y \cdot e^{-x^2-y^2}$")

# - Se dibuja la figura entera.
plt.show()
