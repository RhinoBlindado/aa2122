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

# NOTA: - Este IDE permite tener celdas de código que se pueden ejecutar 
#       independientemente del código que tienen antes o después, 
#       es decir, es como una celda de código de un Colab Notebook: 
#       no es necesario ejecutar todo el programa entero, se puede 
#       ir paso a paso.

#       - Al igual que un Colab Notebook, se han de ejecutar las celdas 
#       en orden la primera vez que se carga el fichero para tener las 
#       funciones y variables de celdas anteriores en memoria. 
#       Una vez realizado esto se pueden ejecutar las celdas en cualquier orden.

#       - Las celdas se delimitan con un comentario de la forma '#%%' y pueden 
#       ejecutarse presionando Ctrl+Enter en una celda resaltada o bien 
#       haciendo clic en el icono que está a la derecha del icono de "Play" 
#       que ejecuta el código entero secuencialmente.

#       - Las celdas están organizadas de manera que la primera celda abarca 
#       todas las funciones implementadas, y luego existe una celda por 
#       cada ejercicio, de esta manera solamente es necesario ejecutar la 
#       celda inicial, denominada la celda 0 y la celda del ejercicio 
#       que se desee ejecutar, el cual se encuentra apropiadamente 
#       identificada en el código.
    
# LIBRERIAS

# - Importando Numpy.
import numpy as np

# - Importanto el dataset de Iris.
from sklearn.datasets import load_iris

# - Importando Matplotlib.
import matplotlib.pyplot as plt

# - Importando para hacer uso de funciones matemáticas más complejas.
import math

# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 2

def trainTestSplit(x_input, y_input, split, className = [None]):
    """
    Realizar la división de los datos en Entrenamiento y Test dado una
    proporcion entr 0 y 1.

    Parameters
    ----------
    x_input : Numpy Array
        Datos de entrada
    y_input : Numpy Array
        Etiquetas de los datos de entrada.
    split : Float
        Proporción de la división
    className : Python List, optional
        Para imprimir los datos por pantalla, por defecto es [None] y 
        no se imprime.

    Returns
    -------
    x_train : Numpy Array
        Datos de entrenamiento.
    y_train : Numpy Array
        Etiquetas de los datos de entrenamiento.
    x_test : Numpy Array
        Datos de test.
    y_test : Numpy Array
        Etiquetas de los datos de test.

    """
    
    # - Obtener las clases del vector de etiquetas, son los números únicos
    # que están, se obtienen con la función 'unique'.
    classes = np.unique(y_input)
        
    # - Se obtiene el tamaño final de los vectores de test y train,
    # en proporción.
    # - Para que no haya problemas con el redondeo, se utiliza floor.
    # y el restante para la otra clase.
    numTrain = math.floor(len(x_input) * split)
    numTest = len(x_input) - numTrain
    
    # Se generan los arrays de ceros que contendrán los datos divididos.
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
        # x_train y x_test, en este momento estarán en orden por cada clase.
        # Después que termine el bucle se volverán a barajear.
        
        startTrain = i * trainDiv
        endTrain = startTrain + trainDiv
        
        startTest = i * testDiv
        endTest = startTest + testDiv

        # Como los datos han sido ya barajeados in-place, se puede obtener
        # los primeros datos hasta la proporción requerida y se añaden al
        # conjunto de entrenamiento, y el resto al conjunto de test 
        # pues ya no están en el orden original.
        
        x_train[startTrain:endTrain] = tempClass[:trainDiv]
        x_test[startTest:endTest] = tempClass[trainDiv:]
        
        # Como el bucle es por cada clase, a cada dato le corresponderá la 
        # clase i-ésima.
        
        y_train[startTrain:endTrain] = i
        y_test[startTest:endTest] = i
        
        # Si se ingresa una lista, se pone en modo verbose.
        if className[0] != None:
            
            print("--- Clase",className[i],"---")
            print("Ejemplos train:", len(tempClass[:trainDiv]))
            print("Ejemplos test:", len(tempClass[trainDiv:]))
        
    # Se obtienen índices con una permutación para volver a barajear los 
    # datos otra vez.
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

# Creando una figura
plt.figure()

# Haciendo una lista de los colores para el Scatter Plot.
colors = ['r', 'g', 'b']

# Por cada etiqueta del dataset...
for i, label in enumerate(iris.target_names.tolist()):
    # ... obtener 'x' e 'y' para cada etiqueta, para eso se utiliza np.where
    # para obtener los indices que tienen las etiquetas i-ésimas, luego se
    # se asigna la columna 0 o 1 a la variable correspondiente de los datos
    # que tienen esa etiqueta...
    x = x_featReduced[ np.where(y_labels == i) ][:, 0]
    y = x_featReduced[ np.where(y_labels == i) ][:, 1]
    
    # ... pintar el Scatter Plot con los datos actuales.
    plt.scatter(x, y, c=colors[i], label=label)
    
# Se imprime la leyenda, ya se infiere de los datos que tiene el Scatter Plot.
plt.legend()

# Renombrando los ejes de la figura acorde a los datos.
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])

# Mostrar la figura.
plt.show()


#%% EJERCICIO 2
###############

# 2.1: Separar en training (80 % de los datos) y test (20 %) aleatoriamente, 
# conservando la proporción de elementos en cada clase tanto en training 
# como en test.

# Cargar el dataset nuevamente.
iris = load_iris()

# Obtener los datos y sus etiquetas.
x_feat = iris.data
y_labels = iris.target

# Llamar a la función personalizada que divide el dataset con 80/20 datos
# para entrenamiento y test.
x_train, y_train, x_test, y_test = trainTestSplit(x_feat, y_labels, 0.8, 
                                                  iris.target_names)

# Imprimir por pantalla los datos.
print("Clase de los ejemplos de entrenamiento:", y_train)
print("Clase de los ejemplos de test:", y_test)

#%% EJERCICIO 3
###############

# 3.1: Obtener 100 valores equiespaciados entre 0 y 4pi
values = np.linspace(0, math.pi * 4, 100)

# 3.2: Obtener el valor de 10^−5 · sinh(x), cos(x) y 
# tanh(2 · sin(x) − 4 · cos(x)) para los 100 valores anteriormente calculados.

#    10^−5 · sinh(x)
hyperSine = math.pow(10, -5) * np.sinh(values)
#   cos(x)
cosine = np.cos(values)
#   tanh(2 · sin(x) − 4 · cos(x)) 
hyperTanSinCos = np.tanh(2 * np.sin(values) - 4 * np.cos(values))

# 3.3: Visualizar las tres curvas simultáneamente en el mismo plot
plt.figure()

# Se tiene la tupla (x, y, formatoLinea), se indica '--' para las lineas 
# punteadas con los colores que se piden.
plot = plt.plot(values, hyperSine, 'g--', values, cosine, 'k--', values, hyperTanSinCos, 'r--')
# Se inserta la leyenda en orden con las tuplas anteriores.
plt.legend(labels=["y = 1e-5 * sinh(x)", "y = cos(x)", "y = tanh(2sin(x) - 4cos(x)"], fontsize=8)
plt.show()

#%% EJERCICIO 4
###############

# 4.1: Funciones

# f(x, y) = 1 - |x + y| - |y - x|
#   Obteniendo los datos en los intervalos pedidos.
x1 = np.linspace(-6, 6, 50)
y1 = np.linspace(-6, 6, 50)

#   Obtener una malla con los datos
x1, y1 = np.meshgrid(x1, y1)

#   Obtener el cálculo pedido.
z1 = 1 - np.abs(x1 + y1) - np.abs(y1 - x1)
    
# f(x, y) = x * y * e^(-x^2-y^2)
#   Ídem con el cálculo anterior.
x2 = np.linspace(-2, 2, 50)
y2 = np.linspace(-2, 2, 50)

x2, y2 = np.meshgrid(x2, y2)

z2 = x2 * y2 * np.exp(-np.power(x2, 2)-np.power(y2,2))

# 4.2: Dibujar en 3D las funciones calculadas.

# Generar la figura, se añade el tamaño en pulgadas y su DPI.
fig = plt.figure(figsize=(15, 10), dpi=300)

# - Como son dos figuras, se añaden dos subplots, se indica con el 2. 
# - Como es la  primera figura, se utiliza el 1 al final.
ax = fig.add_subplot(1, 2, 1, projection='3d')

# Se dibuja la figura con los elementos calculados.
surf = ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

# Añadiendo el título.
plt.title("Pirámide")

# Se añade el segundo subplot, se indica ahora con el 2.
ax = fig.add_subplot(1, 2, 2, projection='3d')

# Se dibuja, se cambia el mapa de color a viridis.
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=plt.cm.viridis,
                       linewidth=0, antialiased=False)

# Se delimitan los ejes para que se vea más grande la figura.
ax.set_zlim(-0.15, 0.15)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Titulo
plt.title("$x \cdot y \cdot e^{-x^2-y^2}$")

# Se dibuja la figura entera.
plt.show()
