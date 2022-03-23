#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 1: Búsqueda Iterativa de Óptimos y Regresión Lineal
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Marzo 2022
    
[ENGLISH]

    Practice 1: Iterative Optimum Search and Linear Regression
    Course: Machine Learning
    Author: Valentino Lugli (Github: @RhinoBlindado)
    March 2022
"""

# LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# FUNCIONES AUXILIARES
def stop():
    input(">STOPPED<")


# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 1

def E(u,v):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.power(u * v * np.exp(-np.power(u, 2)-(np.power(v, 2))), 2)   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return -2 * u * (2 * np.power(u, 2) - 1) * np.power(v, 2) * np.exp(-2 * (np.power(u, 2) + np.power(v, 2)))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return -2 * np.power(u, 2) * v * (2 * np.power(v, 2) - 1) * np.exp(-2 * (np.power(u, 2) + np.power(v, 2)))

#Gradiente de E
def gradE(u,v):
    """
    

    Parameters
    ----------
    u : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(w_ini, lr, grad_fun, fun, epsilon, max_iters):
    """
    Algoritmo de Gradiente Descendente básico

    Parameters
    ----------
    w_ini : TYPE
        DESCRIPTION.
    lr : TYPE
        DESCRIPTION.
    grad_fun : TYPE
        DESCRIPTION.
    fun : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    max_iters : TYPE
        DESCRIPTION.

    Returns
    -------
    w : TYPE
        DESCRIPTION.
    iterations : TYPE
        DESCRIPTION.

    """
        
    w_hist = []
    w = w_ini
    iterations = 0
        
    while(iterations < max_iters and epsilon < fun(w[0], w[1])):
        w = w - lr * grad_fun(w[0], w[1])
        w_hist.append(fun(w[0], w[1]))
        iterations += 1 
        
    return w, iterations, w_hist

def display_figure(rng_val, fun, ws, colormap, title_fig):
    '''
    Esta función muestra una figura 3D con la función a optimizar junto con el 
    óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
    que las otras incluidas en este documento, sirven solamente como referencia y
    apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
    como se prefiera. 
        rng_val: rango de valores a muestrear en np.linspace()
        fun: función a optimizar y mostrar
        ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                               en su búsqueda iterativa del óptimo)
        colormap: mapa de color empleado en la visualización
        title_fig: título superior de la figura
        
    Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
    '''
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    from mpl_toolkits.mplot3d import Axes3D
    x = np.linspace(-rng_val, rng_val, 50)
    y = np.linspace(-rng_val, rng_val, 50)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y) 
    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], fun(ws[:-1,0], ws[:-1,1]), 'b*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'b*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('E(u,v)')

def f(x, y):
    return np.power(x, 2) + 2 * np.power(y, 2) + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)

def dfX(x, y):
    return 2 * (2 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * y) + x)

def dfY(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y) + 4 * y

def gradF(x, y):
    return np.array([dfX(x, y), dfY(x, y)])

def graphError(it, hist_1, hist_2):
    values = range(0, it)
    
    plt.figure()
    plot = plt.plot(values, hist_1, 'r-', values, hist_2, 'b--')
    # Se inserta la leyenda en orden con las tuplas anteriores.
    plt.legend(labels=["$\eta=0.01$", "$\eta=0.1$"], fontsize=8)
    plt.title("Comparación de $\eta$ para función $f(x,y)$")
    plt.show()

    
## FUNCIONES EJER 1 FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1
###############

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

#   1.1: 
#   - Implementado en la función 'gradient_descent' más arriba.

#   1.2:
#       1.2.a:
#       - La gradiente de la función es la derivada, está definida en las
#       funciones 'gradE', 'dEu' y 'dEv'.

# Definiendo diferentes variables para el cálculo:
# - La tasa de aprendizaje pedida
eta = 0.1 

# - Máximo número de iteraciones, el que venía en plantilla.
maxIter = 10000000000

# - El valor al que queremos llegar, sería el "error".
error2get = 1e-8

# - Punto inicial que se indica.
initial_point = np.array([0.5,-0.5])

# Llamando a la función, pasándole el punto inicial, la tasa de aprendizaje,
# la función del gradiente, la función de error (aunque en este caso es la 
# función original), el error a obtener y el máximo de iteraciones.
w, it, _ = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter)

#       1.2.b y 1.2.c:
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")
#%%
# 1.3:
    
#   1.3.a

iniPoint = [-1, 1]
eta = 0.01
maxIter = 50
error = -999

w, it, hist_1 = gradient_descent(iniPoint, eta, gradF, f, error, maxIter)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

iniPoint = [-1, 1]
eta = 0.1
maxIter = 50

w, it, hist_2 = gradient_descent(iniPoint, eta, gradF, f, error, maxIter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

graphError(it, hist_1, hist_2)

# input("\n--- Pulsar tecla para continuar ---\n")

#%%
#   1.3.b

startPoint = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
learningRates = [0.01, 0.1]
error = -999
maxIter = 50


for i in startPoint:
    for j in learningRates:
       w, it, hist = gradient_descent(i, j, gradF, f, error, maxIter)
       print("Punto Inicial: ", i)
       print("Tasa Aprendizaje:",j)
       print("\t(x,y): ", w)
       print("\tf(x,y): ", f(w[0], w[1]))
       print("-------------------------------------")


input("\n--- Pulsar tecla para continuar ---\n")

#%% EJERCICIO 2
###############
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(x_train, y_train, ):
    pass
    return w

# Pseudoinversa	
def pseudoinverse(a):
    pass
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


# w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

#Seguir haciendo el ejercicio...


#%% BONUS
