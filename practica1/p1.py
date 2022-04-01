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
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(1)

# FUNCIONES AUXILIARES
def stop():
    input("\n--- Pulsar tecla para continuar ---\n")

def display_figure(rng_val, fun, ws, colormap, title_fig, elev=None, azimuth=None):
    
    fig = plt.figure(dpi=250)
    ax = Axes3D(fig, auto_add_to_figure=False, computed_zorder=False)
    fig.add_axes(ax)
    plt.title(title_fig)

    x = np.linspace(-rng_val, rng_val, 50)
    y = np.linspace(-rng_val, rng_val, 50)
    
    #   Obtener una malla con los datos
    X, Y = np.meshgrid(x, y)
    
    #   Obtener el cálculo pedido.
    Z = fun(X, Y)
    
    
    if(len(ws) > 0):
        ws = np.asarray(ws)
        ax.plot(ws[0,0], ws[0,1], fun(ws[0,0], ws[0,1]), color='blue', 
                marker='x', markeredgewidth=3, markersize=15, label="Inicio")
        ax.plot(ws[:,0], ws[:,1], fun(ws[:,0], ws[:,1]), color='black')
        ax.plot(ws[-1,0], ws[-1,1], fun(ws[-1,0], ws[-1,1]), color='red', 
                marker='x', markeredgewidth=3, markersize=15, label="Fin")



    ax.plot_surface(X, Y, Z, edgecolor='None', linewidth=.5, rstride=1,
                            cstride=1, cmap=colormap, alpha=.8)

    ax.view_init(elev, azimuth)
    plt.xlim([-rng_val, rng_val])
    plt.ylim([-rng_val, rng_val])
    plt.legend()
    plt.show()
    
    
def display_figure_multi(rng_val, fun, wsArr, labels, colormap, title_fig, elev=None, azimuth=None):

    fig = plt.figure(dpi=250)
    ax = Axes3D(fig, auto_add_to_figure=False, computed_zorder=False)
    fig.add_axes(ax)
    plt.title(title_fig)

    x = np.linspace(-rng_val, rng_val, 50)
    y = np.linspace(-rng_val, rng_val, 50)
    
    #   Obtener una malla con los datos
    X, Y = np.meshgrid(x, y)
    
    #   Obtener el cálculo pedido.
    Z = fun(X, Y)
    
    colors = ['blue', 'red', 'cyan', 'magenta', 'green']
    for i, ws in enumerate(wsArr):
        ws = np.asarray(ws)
        ax.plot(ws[0,0], ws[0,1], fun(ws[0,0], ws[0,1]), color=colors[i], 
                marker='o', markeredgewidth=2, markersize=5, label=labels[i])
        ax.plot(ws[:,0], ws[:,1], fun(ws[:,0], ws[:,1]), color=colors[i])
        ax.plot(ws[-1,0], ws[-1,1], fun(ws[-1,0], ws[-1,1]), color=colors[i], 
                marker='x', markeredgewidth=3, markersize=15)


    ax.plot_surface(X, Y, Z, edgecolor='None', linewidth=.5, rstride=1,
                            cstride=1, cmap=colormap, alpha=.8)

    ax.view_init(elev, azimuth)
    plt.xlim([-rng_val, rng_val])
    plt.ylim([-rng_val, rng_val])
    plt.legend()
    plt.show()


# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 1

def E(u,v):
    """
    Función E(u,v) original

    """
    # Adaptado directamente del guión.
    return np.power(u * v * np.exp(-np.power(u, 2)-(np.power(v, 2))), 2)   

def dEu(u,v):
    """
    Derivada parcial de E con respecto a u

    """
    return -2 * u * (2 * np.power(u, 2) - 1) * np.power(v, 2) * np.exp(-2 * (np.power(u, 2) + np.power(v, 2)))
    
def dEv(u,v):
    """
    Derivada parcial de E con respecto a v

    """
    return -2 * np.power(u, 2) * v * (2 * np.power(v, 2) - 1) * np.exp(-2 * (np.power(u, 2) + np.power(v, 2)))

def gradE(u,v):
    """
    Gradiente de la función E(u, v)

    """
    return np.array([dEu(u,v), dEv(u,v)])

def F(x, y):
    """
    Función f(x,y) original.

    """
    # Adaptado directamente del guión
    return np.power(x, 2) + 2 * np.power(y, 2) + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)

def dfX(x, y):
    """
    Derivada parcial de f con respecto a x.
    
    """
    return 2 * (2 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * y) + x)

def dfY(x, y):
    """
    Derivada parcial de f con respecto a y.

    """
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y) + 4 * y

def gradF(x, y):
    """
    Gradiente de f(x,y)

    """
    return np.array([dfX(x, y), dfY(x, y)])


def gradient_descent(w_ini, lr, grad_fun, fun, epsilon, max_iters):
    """
    Implementación del algoritmo de Gradiente Descendente básico

    Parameters
    ----------
    w_ini : Array
        Pesos iniciales
    lr : Float
        Tasa de aprendizaje o Eta
    grad_fun : Function
        Función de Gradiente, es decir, derivada parcial de E_in.
    fun : Function
        Función original, utilizada como E_in.
    epsilon : Float
        Error que se desea obtener.
    max_iters : Int
        Máximo número de iteraciones.

    Returns
    -------
    w : Array
        Pesos obtenidos luego de llegar al error pedido o en su caso, al máximo
        número de iteraciones
    iterations : Int
        Número de iteraciones para obtener los pesos W
    err_hist: Array
        Traza del valor del error de los pesos en cada iteración del algoritmo.
    w_hist: Array
        Traza del valor de los pesos en cada iteración del algoritmo.

    """
        
    # Crear el vector para guardar la traza del error.
    err_hist = []
    
    # Crear el vector para guardar la traza de los pesos.
    w_hist = []
    
    # Inicializar w a los valores iniciales.
    w = w_ini
    
    # Inicializar el contador.
    iterations = 0
    
    w_hist.append(w_ini)
    err_hist.append(fun(w[0], w[1]))
    
    
    # Mientras se esté dentro de las iteraciones pedidas...
    while(iterations < max_iters):
        # ...Obtener el valor del gradiente, dado esto, la tasa de aprendizaje
        # y los pesos actuales; actualizar dichos pesos a los valores nuevos.
        #   - Esto se puede hacer directamente con las matrices de Numpy 
        #   sin tener que hacer uso de los índices directamente del peso.
        w = w - lr * grad_fun(w[0], w[1])
        
        # Guardar los pesos actuales y su error
        w_hist.append(w)
        err_hist.append(fun(w[0], w[1]))
        
        # Contar una iteración.
        iterations += 1 
        
        # Si se ha llegado a un valor más pequeño o igual que el epsilon pedido,
        # salir del bucle.
        if(epsilon >= fun(w[0], w[1])):
            break
        
    # Retornar los valores pertinentes.
    return w, iterations, err_hist, w_hist


def graph_ex1_3_a(it, hist_1, hist_2, title=""):
    """
    Dibuja las curvas de aprendizaje dependiendo del Eta utilizad.
    
    Función auxiliar para dibujar el gráfico del Ejercicio 1.3.a

    Parameters
    ----------
    it : Int
        Máximo número de iteraciones
    hist_1 : Array
        Traza 1 del gradiente descendente
    hist_2 : Array
        Traza 2 del gradiente descendente
    title : String, optional
        Título al gráfico.

    Returns
    -------
    None.

    """
    
    # Obtener los valores entre [0, it)
    values = range(0, it+1)
    
    plt.figure()
    # Dibujar los gráficos pernitnentes.
    plt.plot(values, hist_1, 'r-', values, hist_2, 'b--')
    # Se inserta la leyenda en orden con las tuplas anteriores.
    plt.legend(labels=["$\eta=0.01$", "$\eta=0.1$"], fontsize=8)
    plt.title(title)
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


def Err(x, y, w):
    """
    Calcula el error medio cuadrático de las entradas x con los pesos w
    donde se utiliza y para medir el error de predicción.

    Parameters
    ----------
    x : Array
        Vector con los valores de la muestra
    y : Array
        Vector con los valores de las etiquetas
    w : Array
        Vector con los valores de los pesos.

    Returns
    -------
    Float
        Valor medio cuadrático de los pesos w

    """
    
    # Adaptación directa de la fórmula matricial
    h_w = x.dot(w)
    return np.mean( np.square(h_w - y) )


def gradSGD(w, x, y):
    """
    Gradiente del error medio cuadrático

    Parameters
    ----------
    x : Array
        Vector con los valores de la muestra
    y : Array
        Vector con los valores de las etiquetas
    w : Array
        Vector con los valores de los pesos.

    Returns
    -------
    Array
        Vector con los pesos modificicado

    """
    
    # Adaptación directa de la fórmula matricial
    h_w = x.dot(w)
    return (2/len(x)) * ( x.T.dot( h_w - y ) )


def sgd(x, y, wIni, lr, batchSize, epsilon, maxIters):
    """
    
    Implementación del algoritmo de Gradiente Descendente Estocástico por
    Minibatches
    
    Parameters
    ----------
    x : Array
        Los datos de entrada.
    y : Array
        Las etiquetas de los datos de entrada.
    wIni : Array
        Los pesos iniciales
    lr : Float
        La tasa de aprendizaje o Eta
    batchSize : Int
        Tamaño del batch de entrenamiento.
    epsilon : Float
        Error mínimo a obtener.
    maxIters : Int
        Máximo número de iteraciones.

    Returns
    -------
    w : TYPE
        DESCRIPTION.
    i : TYPE
        DESCRIPTION.

    """
    
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
            i += 1 
            
            if(i < maxIters): break
            
        if(Err(x, y, w) < epsilon):
            break
        
    return w, i
        
def pseudoinverse(x, y):
    """
    
    Implementación del algoritmo "Normal"/Pseudoinversa
    
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
    
    yNoided = np.copy(y)
    
    noiseSize = int(len(y) * (percent / 2))

    posNoise = np.where(y == 1)
    negNoise = np.where(y == -1)

    np.random.shuffle(posNoise[0])
    np.random.shuffle(negNoise[0])
    
    posNoise = posNoise[0][:noiseSize]
    negNoise = negNoise[0][:noiseSize]
    
    yNoided[posNoise] = -1
    yNoided[negNoise] = 1

    return yNoided

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
    

def graph_ex2_2(xData, yData, yLabel, w, title=None, showFit=True, nonLinear=False):
    plt.figure()
    
    # Haciendo una lista de los colores para el Scatter Plot.
    colors = ['k', 'y']


    # Por cada etiqueta del dataset...
    for i, label in enumerate(yLabel):
        x = xData[ np.where(yData == label) ][:, 1]
        y = xData[ np.where(yData == label) ][:, 2]
        
        # ... pintar el Scatter Plot con los datos actuales.
        plt.scatter(x, y, c=colors[i])
    
    if(showFit):
        # Dibujando encima las rectas que dividen el hiperplano en 2:
        x1 = np.linspace(-1, 1, len(yData))
        
        if(not nonLinear):
            x2 = (-w[0] - w[1] * x1) / w[2]    
            plt.plot(x1, x2, 'b-')
        else:
            x2 = np.linspace(-1, 1, len(yData))
            X1, X2 = np.meshgrid(x1, x2)
            ZW = w[0] + (w[1] * X1 )+ (w[2] * X2) + (w[3] * X1 * X2 )+ (w[4] * np.square(X1)) + (w[5] * np.square(X2))
            
            plt.contour(X1, X2, ZW, 0, colors='b')
            
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    # Se imprime la leyenda, ya se infiere de los datos que tiene el Scatter Plot.
    plt.title(title)
    plt.show()

def acc(x, y, w):
    """
    Obtener el 'accuracy' del modelo.
    
    Calculado como Nº Aciertos / Nº Ejemplos

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    w : TYPE
        DESCRIPTION.

    Returns
    -------
    perc : TYPE
        DESCRIPTION.

    """
    # Se realiza una "predicción" de los datos.
    h_w = x.dot(w)
    
    # Inicializando el contador de aciertos.
    good = 0
    
    for i, j in enumerate(y):
        # Si la predicción tiene el mismo signo que la etiqueta, se cataloga
        # como que ha acertado.
        if((h_w[i] >= 0 and j > 0) or
           (h_w[i] < 0 and j < 0)):
            good += 1
            
    # Se obtiene una proporción entre [0, 1] y se retorna.
    perc = good / len(y)

    return perc
    

## FUNCIONES EJER 2 FIN

## FUNCIONES BONUS

def hessian(x,y):
    hess = np.array([[2 - 8 * np.square(np.pi) * np.sin(2 * np.pi * x) * np.sin(np.pi * y), 4 * np.square(np.pi) * np.cos(2 * np.pi * x) * np.cos(np.pi * y)],
                     [4 * np.square(np.pi) * np.cos(2 * np.pi * x) * np.cos(np.pi * y), 4 - 2 * np.square(np.pi) * np.sin(2 * np.pi * x) * np.sin(np.pi * y)]])
    
    return hess

def newton(w_ini, lr, grad_fun, fun, epsilon, max_iters):
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
        
    errHist = []
    wHist= []
    w = w_ini
    iterations = 0
    
    wHist.append(w)
    errHist.append(fun(w[0], w[1]))

    while(iterations < max_iters):
        
        hess = hessian(w[0], w[1])
        w = w - lr * np.dot(np.linalg.inv(hess), grad_fun(w[0], w[1]).T)
        
        
        errHist.append(fun(w[0], w[1]))
        wHist.append(w)

        iterations += 1 
        
        if(epsilon > fun(w[0], w[1])):
            break;
        
    return w, iterations, wHist, errHist


def graph_bonus_1(it, hGD1, hGD2, hNewt1, hNewt2, title=""):
    values = range(0, it+1)
    
    plt.figure()
    plt.plot(values, hGD1, 'r-', label="GD, $\eta=0.01$")
    plt.plot(values, hGD2, 'r--', label="GD, $\eta=0.1$")
    plt.plot(values, hNewt1, 'b-', label="Newton, $\eta=0.01$")
    plt.plot(values, hNewt2, 'b--', label="Newton, $\eta=0.1$")


    # Se inserta la leyenda en orden con las tuplas anteriores.
    # plt.legend(labels=["$\eta=0.01$", "$\eta=0.1$"], fontsize=8)
    plt.legend()

    plt.title(title)
    plt.show()
    
    
def graph_bonus_2(it, hist, pts, title=""):
    values = range(0, it)
    
    plt.figure()
    
    for i, h in enumerate(hist):
        plt.plot(values, h, 'r-')


    # Se inserta la leyenda en orden con las tuplas anteriores.
    # plt.legend(labels=["$\eta=0.01$", "$\eta=0.1$"], fontsize=8)
    plt.legend()

    plt.title(title)
    plt.show()

## FUNCIONES BONUS FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1.2
#################

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
w, it, _, w_hist = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter)


#       1.2.b y 1.2.c:
print ('- Numero de iteraciones: ', it)
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')')
print ("- Valor E(u,v) en punto:", E(w[0], w[1]))

display_figure(2, E, w_hist, 'coolwarm',"Evolución del Gradiente $E(u,v)$", azimuth=100)

# 1.3:

#%% EJERCICIO 1.3.a
###################

print('Apartado 1.3.a')

# Por consistencia se redefinen variables para cada llamada del gradiente
# descendente.

iniPoint = [-1, 1]
eta = 0.01
# - Para evitar que el algoritmo se detenga antes de las 50 iteraciones, se fija
# el error a un valor inalcanzable.
error = -999
maxIter = 50

# Ejecudanto GD son eta=0.01
w, it, hGD1, wHist1 = gradient_descent(iniPoint, eta, gradF, F, error, maxIter)

print("Con eta = 0.01")
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')
print ("- Valor f(x,y) en punto:", F(w[0], w[1]))

# Ejecudanto GD son eta=0.1
eta = 0.1
w, it, hGD2, wHist2 = gradient_descent(iniPoint, eta, gradF, F, error, maxIter)

print("Con eta = 0.1")
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')
print ("- Valor f(x,y) en punto:", F(w[0], w[1]))


print("Imprimiendo gráficas...")
graph_ex1_3_a(it, hGD1, hGD2, title="Evolución de $f(x,y)$ por iteración dependiendo de $\eta$")

display_figure(2, F, wHist1, 'coolwarm',"Evolución del Gradiente de f(x,y), $\eta=0.01$", elev=35, azimuth=230)
display_figure(2, F, wHist2, 'coolwarm',"Evolución del Gradiente de f(x,y), $\eta=0.1$", elev=35, azimuth=230)

print("Listo.")

#%%
#   1.3.b
print('Apartado 1.3.b\n')

# Ahora se meten los valores en unas listas para hacer un bucle que
# calcule e imprima cada valor.
startPoint = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
learningRates = [0.01, 0.1]

wHist3 = []

for i in startPoint:
    for j in learningRates:
        w, it, _, wH = gradient_descent(i, j, gradF, F, error, maxIter)
        wHist3.append(wH)
        print("Punto Inicial: ", i)
        print("Tasa Aprendizaje:",j)
        print("\t(x,y): ", w)
        print("\tf(x,y): ", F(w[0], w[1]))
        print("--------------------------")


# Dibujandolo, para que se pueda ver mejor.
display_figure_multi(3.25, F, wHist3[::2], ['(-0.5, -0.5)','(1, 1)','(2.1, -2.1)','(-3, 3)','(-2, 2)'], 'coolwarm', "Diferentes puntos de inicio para $f(x,y)$, $\eta=0.01$", elev=50, azimuth=180)
display_figure_multi(3.25, F, wHist3[::2], ['(-0.5, -0.5)','(1, 1)','(2.1, -2.1)','(-3, 3)','(-2, 2)'], 'coolwarm', "Ídem, vista aérea", elev=90, azimuth=180)

display_figure_multi(3.25, F, wHist3[1::2], ['(-0.5, -0.5)','(1, 1)','(2.1, -2.1)','(-3, 3)','(-2, 2)'], 'coolwarm', "Diferentes puntos de inicio para $f(x,y)$, $\eta=0.1$", elev=50, azimuth=180)
display_figure_multi(3.25, F, wHist3[1::2], ['(-0.5, -0.5)','(1, 1)','(2.1, -2.1)','(-3, 3)','(-2, 2)'], 'coolwarm', "Ídem, vista aérea", elev=90, azimuth=180)

#   1.4: 
#   - En la memoria

#%% EJERCICIO 2.1
#################
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
print (" - E_in: ", Err(x1, y1, wEndPseudo), ", Acc: ", acc(x1, y1, wEndPseudo))
print (" - E_out: ", Err(x_test1, y_test1, wEndPseudo), ", Acc:", acc(x_test1, y_test1, wEndPseudo))

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
print (" - E_in: ", Err(x1, y1, wEndSGD), ", Acc: ", acc(x1, y1, wEndSGD))
print (" - E_out: ", Err(x_test1, y_test1, wEndSGD), ", Acc:", acc(x_test1, y_test1, wEndSGD))

# Se imprime la gráfica con los resultados.

graph_ex2_1(x1, y1, [-1, 1], [1, 5], wEndPseudo, title="Train: Pseudoinversa")
graph_ex2_1(x1, y1, [-1, 1], [1, 5], wEndSGD, title="Train: SGD")

graph_ex2_1(x_test1, y_test1, [-1, 1], [1, 5], wEndPseudo, title="Test: Pseudoinversa")
graph_ex2_1(x_test1, y_test1, [-1, 1], [1, 5], wEndSGD, title="Test: SGD")


#%% EJERCICIO 2.2
#################

print('Apartado 2.2.a-c\n')

# 2.2: 
#   2.2.a:
#   - Generando una muestra de N=1000 puntos en el cuadrado X=[-1, 1]x[-1, 1]

x2 = simula_unif(1000, 2, 1)

#   2.2.b:
#   - Asigando etiquetas a los puntos.
y2 = getTags(x2)

#   - Cambiar el signo a 10% de las etiquetas
y2 = addNoise(y2, 0.1)

#   2.2.c:
#   - Usando como vector de características [1, x1, x2]
x2NonL = phi(x2)

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

print ('\nBondad del resultado para SGD con características no lineales:\n')
print (" - E_in: ", Err(x2, y2, wEnd), ", Acc: ", acc(x2, y2, wEnd))

#   - Mostrar el error
graph_ex2_2(x2, y2, [-1, 1], wEnd, title="Datos", showFit=False)
graph_ex2_2(x2, y2, [-1, 1], wEnd, title="Características Lineales")

#%% EJERCICIO 2.2.d
###################
#   2.2.d:
#   - Ejecutar todo el experimento definido por (a)-(c) 1000 veces 
#   (generamos 1000 muestras diferentes)

print("Apartado 2.2.d")

meanEin2d = 0
meanEout2d = 0

meanEinAcc2d = 0
meanEoutAcc2d = 0

print("Iniciando experimento...")
print("0|          |100%\n |", end='')
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
    
    meanEinAcc2d += acc(xTrain2d, yTrain2d, wEnd2d)
    meanEoutAcc2d +=  acc(xTest2d, yTest2d, wEnd2d)
    
    if(i%100 == 0):
        print("█", end='')

print("|")

meanEin2d = meanEin2d / 1000
meanEout2d = meanEout2d / 1000

meanEinAcc2d = meanEinAcc2d / 1000
meanEoutAcc2d = meanEoutAcc2d / 1000

print("Resultados 1000 iteraciones:\n")
print(" - Ein medio:", meanEin2d, ", Acc: ", meanEinAcc2d)
print(" - Eout medio:", meanEout2d, ", Acc:", meanEoutAcc2d)
    
#   2.2.e:
#   - En memoria.

#%% EJERCICIO 2.2.f.1
#####################

#   2.2.f:
#   - Repetir el mismo experimento anterior pero usando características no lineales

print("Apartado 2.2.f.1")

maxIters2f = 200
eta2f = 0.01
error2f = 0
batchSize2f = 32
wIni2f = np.array([0, 0, 0, 0, 0, 0])

meanEin2f = 0
meanEout2f = 0

xTrain2f = simula_unif(1000, 2, 1)

yTrain2f = addNoise(getTags(xTrain2f), 0.1)

#   Añadiendo características no lineales al modelo.
xTrain2f = phi(xTrain2f)

xTrain2f = addBias(x2NonL)

wEnd2f, _ = sgd(xTrain2f, y2, wIni2f, eta2f, batchSize2f, error2f, maxIters2f)


print ('\nBondad del resultado para SGD con características no lineales:\n')
print (" - E_in: ", Err(xTrain2f, y2, wEnd2f), ", Acc: ", acc(xTrain2f, y2, wEnd2f))

#   - Mostrar el error
graph_ex2_2(xTrain2f, y2, [-1, 1], wEnd2f, title="Características No Lineales", nonLinear=True)

#%% EJERCICIO 2.2.f.2
#####################

print("Apartado 2.2.f.2")

meanEin2f = 0
meanEout2f = 0

meanEinAcc2f = 0
meanEoutAcc2f = 0

print("Iniciando experimento...")
print("0|          |100%\n |", end='')
for i in range(1000):
    xTrain2f = simula_unif(1000, 2, 1)
    xTest2f = simula_unif(1000, 2, 1)
    
    yTrain2f = addNoise(getTags(xTrain2f), 0.1)
    yTest2f = addNoise(getTags(xTest2f), 0.1)
    
    xTrain2f = phi(xTrain2f)
    xTest2f = phi(xTest2f)

    xTrain2f = addBias(xTrain2f)
    xTest2f = addBias(xTest2f)
    
    wEnd2f, _ = sgd(xTrain2f, yTrain2f, wIni2f, eta2f, batchSize2f, error2f, maxIters2f)
    
    meanEin2f += Err(xTrain2f, yTrain2f, wEnd2f)
    meanEout2f += Err(xTest2f, yTest2f, wEnd2f)
    
    meanEinAcc2f += acc(xTrain2f, yTrain2f, wEnd2f)
    meanEoutAcc2f +=  acc(xTest2f, yTest2f, wEnd2f)

    if(i%100 == 0):
        print("█", end='')

print("|")

print("Experimento completado.")

meanEin2f = meanEin2f / 1000
meanEout2f = meanEout2f / 1000

meanEinAcc2f = meanEinAcc2f / 1000
meanEoutAcc2f = meanEoutAcc2f / 1000

print("\nResultados 1000 iteraciones:")
print(" - Ein medio:", meanEin2f, ", Acc: ", meanEinAcc2f)
print(" - Eout medio:", meanEout2f, ", Acc:", meanEoutAcc2f)
    
#%% BONUS
#########

print("\nBonus: Método de Newton\n")

iniPoint = np.array([-1, 1])
eta = 0.01
error = -999
maxIter = 50

w, it, errNewt1, wHistNewt1 = newton(iniPoint, eta, gradF, F, error, maxIter)

print("Con eta = 0.01")
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')

eta = 0.1
w, it, errNewt2, wHistNewt2 = newton(iniPoint, eta, gradF, F, error, maxIter)

print("Con eta = 0.1")
print ('- Coordenadas obtenidas: (', w[0], ', ', w[1],')\n')


print("Imprimiendo gráfica...")
graph_bonus_1(it, hGD1, hGD2, errNewt1, errNewt2, title="GD vs Newton: Error por iteración")
print("Listo.")

#%%
startPoint = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
learningRates = [0.01, 0.1]

hist = []

for i in startPoint:
    for j in learningRates:
        w, it, _, h = newton(i, j, gradF, F, error, maxIter)
        hist.append(h)
        print("Punto Inicial: ", i)
        print("Tasa Aprendizaje:",j)
        print("\t(x,y): ", w)
        print("\tf(x,y): ", F(w[0], w[1]))
        print("--------------------------")

#%%
# graph_bonus_2(it, hist, startPoint)
display_figure(2, F, wHistNewt1, 'coolwarm',"Evolución del Gradiente de f(x,y), $\eta=0.01$", elev=35, azimuth=230)

# display_figure_multi(3.25, F, hist[::2], ['(-0.5, -0.5)','(1, 1)','(2.1, -2.1)','(-3, 3)','(-2, 2)'], 'coolwarm', "Diferentes puntos de inicio para $f(x,y)$, $\eta=0.01$", elev=50, azimuth=180)
