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
    input("\n--- Pulsar tecla para continuar ---\n")


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
    
def dEv(u,v):
    """
    Derivada parcial de E con respecto a v

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

def gradE(u,v):
    """
    Gradiente de E

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
    while(iterations < max_iters):
        w = w - lr * grad_fun(w[0], w[1])
        w_hist.append(fun(w[0], w[1]))
        
        iterations += 1 
        
        if(epsilon > fun(w[0], w[1])):
            break;
        
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
    ax = Axes3D(fig, auto_add_to_figure=False)
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

def funcF(x, y):
    return np.power(x, 2) + 2 * np.power(y, 2) + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)

def dfX(x, y):
    return 2 * (2 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * y) + x)

def dfY(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y) + 4 * y

def gradF(x, y):
    return np.array([dfX(x, y), dfY(x, y)])

def graph_ex1_3_a(it, hist_1, hist_2):
    values = range(0, it)
    
    plt.figure()
    plt.plot(values, hist_1, 'r-', values, hist_2, 'b--')
    # Se inserta la leyenda en orden con las tuplas anteriores.
    plt.legend(labels=["$\eta=0.01$", "$\eta=0.1$"], fontsize=8)
    plt.title("Evolución del error por iteración dependiendo de $\eta$")
    plt.show()

## FUNCIONES EJER 1 FIN

## FUNCIONES EJER 2

def readData(file_x, file_y):
    # Funcion para leer los datos
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
def Err(x, y, w):
    h_w = x.dot(w)
    return np.mean( np.square(h_w - y) )


def gradSGD(w, x, y):
    h_w = x.dot(w)
    return (2/len(x)) * ( x.T.dot( h_w - y ) )


# Gradiente Descendente Estocastico
def sgd(x, y, wIni, lr, batchSize, epsilon, maxIters):
    
    i = 0
    w = wIni
    batches = int(np.ceil(len(x) / batchSize))
    
    if (batches < 1): batches = 1 

    while(i < maxIters):
        
        shuffleOrder = np.random.permutation(len(x))
        xShuff = x[shuffleOrder]
        yShuff = y[shuffleOrder]
        
        for j in range(batches): 
            
            ini = j*batchSize
            fin = j*batchSize+batchSize
            
            if(fin > len(x)): fin = len(x)
            
            w = w - lr * gradSGD(w, xShuff[ini:fin], yShuff[ini:fin])
            
        if(Err(x, y, w) < epsilon):
            break
        i += 1 
        
    return w, i
        
def pseudoinverse(x, y):
    """
    

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    w : TYPE
        DESCRIPTION.

    """
    # Obtener los valores singulares de x.
    u, sVect, vT = np.linalg.svd(x)
    
    # Invertir S, ya que Numpy lo que regresa es un vector en vez de una 
    # matriz, se puede invertir directamente con una división.
    sInvVect = 1.0 / sVect
    
    # Crear una matriz cuadrada donde estará S^-1 del tamaño de x.
    sInv = np.zeros(x.shape)
    
    # Rellenar la matriz con la diagonal invertida.
    np.fill_diagonal(sInv, sInvVect)
    
    # Calcular la pseudoinversa, la X^cruz
    xCross = vT.T.dot(sInv.T).dot(u.T)
    
    # Obtener los pesos en un solo paso.
    w = xCross.dot(y)

    return w


def graph_ex2_1(xData, yData, yLabel, yNames, w, title=None):
    plt.figure()
    
    # Haciendo una lista de los colores para el Scatter Plot.
    colors = ['b', 'r']

    # Por cada etiqueta del dataset...
    for i, label in enumerate(yLabel):
        x = xData[ np.where(yData == label) ][:, 1]
        y = xData[ np.where(yData == label) ][:, 2]
        
        # ... pintar el Scatter Plot con los datos actuales.
        plt.scatter(x, y, c=colors[i], label=yNames[i])
    
    
    # Renombrando los ejes de la figura acorde a los datos.
    plt.xlabel("$x_1$ - Intensidad Promedio")
    plt.ylabel("$x_2$ - Simetría")

    # Dibujando encima las rectas que dividen el hiperplano en 2
    xW = np.linspace(0, 1, len(yData))
    yW = (-w[0] - w[1] * xW) / w[2]    
      
    plt.plot(xW, yW, 'c--')

    # Se imprime la leyenda, ya se infiere de los datos que tiene el Scatter Plot.
    plt.legend()
    
    plt.title(title)
    plt.show()


def simula_unif(N, d, size):
    # Simula datos en un cuadrado [-size,size]x[-size,size]
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

def getTags(x):
    y = []
    for i in x:
        y.append(f(i[0], i[1]))
    
    return np.asarray(y)

def addNoise(y, percent):
    noiseSize = int(len(y) * percent)
    noiseIdxs = np.random.permutation(len(y))[:noiseSize]

    for i in noiseIdxs:
        if(y[i] > 0):
            y[i] = -1
        else:
            y[i] = 1

    return y

def addBias(x):
    
    bias = np.ones((x.shape[0],1))
    xBias = np.hstack((bias, x))
    
    return xBias

def phi(x):
    
    nonLinear = np.empty((x.shape[0], 3))
    
    nonLinear[:, 0] = x[:, 0] * x[:, 1]
    nonLinear[:, 1] = np.square(x[:, 0])
    nonLinear[:, 2] = np.square(x[:, 1])
        
    xNonL = np.hstack((x, nonLinear))
    
    return xNonL
    

def graph_ex2_2(xData, yData, yLabel, w, title=None, nonLinear=False):
    plt.figure()
    
    # Haciendo una lista de los colores para el Scatter Plot.
    colors = ['k', 'y']


    # Por cada etiqueta del dataset...
    for i, label in enumerate(yLabel):
        x = xData[ np.where(yData == label) ][:, 1]
        y = xData[ np.where(yData == label) ][:, 2]
        
        # ... pintar el Scatter Plot con los datos actuales.
        plt.scatter(x, y, c=colors[i])
    

    # Dibujando encima las rectas que dividen el hiperplano en 2:
    xW = np.linspace(-1, 1, len(yData))
    
    if(not nonLinear):
        yW = (-w[0] - w[1] * xW) / w[2]    
        plt.plot(xW, yW, 'b-')
    else:
        pass
        
        
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # Se imprime la leyenda, ya se infiere de los datos que tiene el Scatter Plot.
    plt.title(title)
    plt.show()

## FUNCIONES EJER 2 FIN

## FUNCIONES BONUS

def newtonsMethod():
    pass

## FUNCIONES BONUS FIN



# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1
###############

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS')
print('Apartado 1.2')


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
print ('- Numero de iteraciones: ', it)
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')')

stop()

# 1.3:    
    
#   1.3.a
print('Apartado 1.3.a')

# Por consistencia se redefinen variables para cada llamada del gradiente
# descendente.

iniPoint = [-1, 1]
eta = 0.01
error = -999
maxIter = 50

w, it, hist_1 = gradient_descent(iniPoint, eta, gradF, funcF, error, maxIter)

print("Con eta = 0.01")
print ('- Numero de iteraciones: ', it)
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')

eta = 0.1
w, it, hist_2 = gradient_descent(iniPoint, eta, gradF, funcF, error, maxIter)

print("Con eta = 0.1")
print ('- Numero de iteraciones: ', it)
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')


print("Imprimiendo gráfica...")
graph_ex1_3_a(it, hist_1, hist_2)
print("Listo.")

stop()

#   1.3.b
print('Apartado 1.3.b\n')

# Ahora se meten los valores en unas listas para hacer un bucle que
# calcule e imprima cada valor.
startPoint = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
learningRates = [0.01, 0.1]

for i in startPoint:
    for j in learningRates:
        w, it, hist = gradient_descent(i, j, gradF, funcF, error, maxIter)
        print("Punto Inicial: ", i)
        print("Tasa Aprendizaje:",j)
        print("\t(x,y): ", w)
        print("\tf(x,y): ", funcF(w[0], w[1]))
        print("--------------------------")

        
stop()

#   1.4: 
#   - En la memoria

#%% EJERCICIO 2
###############
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Apartado 2.1\n')

# 2.1:
    
# Marcando los labels del ejercicio.
label5 = 1
label1 = -1

# Lectura de los datos de entrenamiento
x1, y1 = readData('datos/X_train.npy', 'datos/y_train.npy')

# Lectura de los datos para el test
x_test1, y_test1 = readData('datos/X_test.npy', 'datos/y_test.npy')

# Se obtienen los pesos con la pseudoinversa en un solo cálculo.
wEndPseudo = pseudoinverse(x1, y1)

print ('Bondad del resultado para pseudoinversa:\n')
print (" - E_in: ", Err(x1, y1, wEndPseudo))
print (" - E_out: ", Err(x_test1, y_test1, wEndPseudo))

# Configuración para SGD:

# - Como recomendación, se indican 200 iteraciones máximas.
maxIters = 200

# - Pesos iniciales
wIni = np.array([0, 0, 0])

# - Tasa de aprendizaje
eta = 0.01

# - Ya el que el algoritmo también puede detenerse a un error, como no se
# especifica un valor de error mínimo, se coloca este valor de manera que
# no llegará nunca al mismo.
error = -999

# - Como recomendación, se coloca un tamaño de batch de 32.
batchSize = 32

# Se realiza el algoritmo de SGD con los parámetros establecidos.
wEndSGD, iters = sgd(x1, y1, wIni, eta, batchSize, error, maxIters)

print ('\nBondad del resultado para grad. descendente estocástico:\n')
print (" - Ein: ", Err(x1, y1, wEndSGD))
print (" - Eout: ", Err(x_test1, y_test1, wEndSGD))

# Se imprime la gráfica con los resultados

graph_ex2_1(x1, y1, [-1, 1], [1, 5], wEndPseudo, title="Pseudoinversa")
graph_ex2_1(x1, y1, [-1, 1], [1, 5], wEndSGD, title="SGD")

stop()

#%%
# 2.2: 
#   2.2.a:
#   - Generando una muestra de N=1000 puntos en el cuadrado X=[-1, 1]x[-1, 1]

x2 = simula_unif(1000, 2, 1)

#   2.2.b:
#   - Asigando etiquetas a los puntos.
y2 = getTags(x2)

#   - Cambiar el signo a 10% de las etiquetas
y2 = addNoise(y2, 0.1)

print("Apartado 2.2.c")
#   2.2.c:
#   - Usando como vector de características [1, x1, x2]
x2 = addBias(x2)

#   - Ajustar un modelo de regresión lineal al conjunto de datos generado 
#   y estimar los pesos w.

# Definiendo los parámetros de SGD:
maxIters = 200
wIni = np.array([0, 0, 0])
eta = 0.01
error = 0
batchSize = 32

#   -  Estimar el error de ajuste Ein usando SGD.
wEnd, iters = sgd(x2, y2, wIni, eta, batchSize, error, maxIters)

print ('\nBondad del resultado para grad. descendente estocástico:\n')
print (" - Ein: ", Err(x2, y2, wEnd))

#   - Mostrar el error
graph_ex2_2(x2, y2, [-1, 1], wEnd, title="Características Lineales")


# stop()

#%%
#   2.2.d:
#   - Ejecutar todo el experimento definido por (a)-(c) 1000 veces 
#   (generamos 1000 muestras diferentes)

print("Apartado 2.2.d")

meanEin2d = 0
meanEout2d = 0

print("Iniciando experimento...")
for i in range(1000):
    xTrain2d = simula_unif(1000, 2, 1)
    xTest2d = simula_unif(1000, 2, 1)
    
    yTrain2d = addNoise(getTags(xTrain2d), 0.1)
    yTest2d = addNoise(getTags(xTest2d), 0.1)
    
    xTrain2d = addBias(xTrain2d)
    xTest2d = addBias(xTest2d)
    
    wIni2d = np.array([0, 0, 0])
    
    wEnd2d, _ = sgd(xTrain2d, yTrain2d, wIni2d, eta, batchSize, error, maxIters)
    
    meanEin2d += Err(xTrain2d, yTrain2d, wEnd2d)
    meanEout2d += Err(xTest2d, yTest2d, wEnd2d)
    
    if(i == 250):
        print("25% completado...")
        
    if(i == 500):
        print("50% completado...")
        
    if(i == 750):
        print("75% completado...")

print("Experimento completado.")

meanEin2d = meanEin2d / 1000
meanEout2d = meanEout2d / 1000

print("Resultados 1000 iteraciones:\n")
print(" - Ein medio:", meanEin2d)
print(" - Eout medio:", meanEout2d)
    
stop()

#   2.2.e:
#   - En memoria.

#%%
#   2.2.f:
#   - Repetir el mismo experimento anterior pero usando características no lineales

print("Apartado 2.2.f")

maxIters2f = 200
eta2f = 0.01
error2f = 0
batchSize2f = 32
wIni2f = np.array([0, 0, 0, 0, 0, 0])

meanEin2f = 0
meanEout2f = 0

print("Iniciando experimento...")

for i in range(1000):
    xTrain2f = simula_unif(1000, 2, 1)
    xTest2f = simula_unif(1000, 2, 1)
    
    yTrain2f = addNoise(getTags(xTrain2f), 0.1)
    yTest2f = addNoise(getTags(xTest2f), 0.1)
    
#   Añadiendo características no lineales al modelo.
    xTrain2f = phi(xTrain2f)
    xTest2f = phi(xTest2f)

    xTrain2f = addBias(xTrain2f)
    xTest2f = addBias(xTest2f)

    wEnd2f, _ = sgd(xTrain2f, yTrain2f, wIni2f, eta2f, batchSize2f, error2f, maxIters2f)
    
    meanEin2f += Err(xTrain2f, yTrain2f, wEnd2f)
    meanEout2f += Err(xTest2f, yTest2f, wEnd2f)

    if(i == 0):
        graph_ex2_2(xTrain2f, yTrain2f, [-1, 1], wEnd2f, title="Características No Lineales $\phi_2(x)$", nonLinear=True)

    if(i == 250):
        print("\t25% completado...")
        
    if(i == 500):
        print("\t50% completado...")
        
    if(i == 750):
        print("\t75% completado...")

print("Experimento completado.")

meanEin2f = meanEin2f / 1000
meanEout2f = meanEout2f / 1000

print("\nResultados 1000 iteraciones:")
print(" - Ein medio:", meanEin2f)
print(" - Eout medio:", meanEout2f)
    
stop()

#%% BONUS

print("\nBonus: Método de Newton\n")


