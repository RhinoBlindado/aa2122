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
    """
    Función auxiliar sin utilizar; puede añadirse donde se vea necesario.

    """
    input("--Ejecución pausada--")

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

    # Variante para dibujar múltiples rutas
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


def graph_ex1_3_a(it, hist_1, hist_2, labels, title=""):
    """
    Dibuja las curvas de aprendizaje dependiendo del Eta utilizad.
    
    Función auxiliar para dibujar el gráfico del Ejercicio 1.3.a

    """
    
    # Obtener los valores entre [0, it)
    values = range(0, it+1)
    
    plt.figure()
    # Dibujar los gráficos pernitnentes.
    plt.plot(values, hist_1, 'r-', values, hist_2, 'b--')
    # Se inserta la leyenda en orden con las tuplas anteriores.
    plt.legend(labels=labels, fontsize=8)

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
    w : Array
        Los pesos calculados luego de las iteraciones.
    i : Int
        Las iteraciones que tomó el algoritmo.

    """
    # Inicializar el contador y los pesos.
    i = 0
    w = wIni
    
    # Determinar cuantos lotes hay que hacer, para esto se divide la cantidad
    # de datos que se tienen por el tamaño del lote deseado, así se obtiene
    # la cantidad de lotes. Para evitar decimales, se toma la función ceil()
    batches = int(np.ceil(len(x) / batchSize))
    
    # Si por alguna razón sucede que son menos de 1, se fija a 1 los lotes.
    if (batches < 1): batches = 1 

    # Mientras no se haya llegado al límite de iteraciones...
    while(i < maxIters):
        
        # ...Obtener un vector de índices de tamaño x sin repetir...
        shuffleOrder = np.random.permutation(len(x))
        # ...y desordenar de esta manera los datos, manteniendo la correspondencia
        # entre dato y etiqueta...
        xShuff = x[shuffleOrder]
        yShuff = y[shuffleOrder]
        
        # ...y para todos los lotes...
        for j in range(batches): 
            
            # ...Obtener donde comienza y termina.
            ini = j*batchSize
            fin = j*batchSize+batchSize
            
            # Si se sobrepasa de la longitud de los datos, se acorta a esa.
            if(fin > len(x)): fin = len(x)
            
            # Realizar el cálculo del gradiente con el lote j-ésimo.
            w = w - lr * gradSGD(w, xShuff[ini:fin], yShuff[ini:fin])
            
            # Sumar una iteración.
            i += 1 
            
            # Si se obtiene el máximo de iteraciones dentro de este bucle, 
            # cortar la ejecución también.
            if(i < maxIters): break
            
        # Si el error después de una época es menor al que se desea, también
        # finalizar.
        if(Err(x, y, w) < epsilon):
            break
        
    return w, i
        
def pseudoinverse(x, y):
    """
    
    Implementación del algoritmo "Normal"/Pseudoinversa
    
    Parameters
    ----------
    x : Array
        Los datos de entrada.
    y : Array
        Las etiquetas de los datos de entrada.

    Returns
    -------
    w : Array
        Los pesos calculados luego de las iteraciones.
    """
    # Obtener los valores singulares de x.
    u, sVect, vT = np.linalg.svd(x)
    
    # Invertir S, ya que Numpy lo que regresa es un vector en vez de una 
    # matriz, se puede invertir directamente con una división.
    sInvVect = 1.0 / sVect
    
    # Crear una matriz cuadrada donde estará S^-1 del tamaño de x.
    sInv = np.zeros(x.shape)
    
    # Rellenar la matriz con los valores, luego deberá de trasponerse esta
    # matriz para que funcione bien.
    np.fill_diagonal(sInv, sInvVect)
    
    # Calcular la pseudoinversa, la X^cruz
    xCross = vT.T.dot(sInv.T).dot(u.T)
    
    # Obtener los pesos en un solo paso.
    w = xCross.dot(y)

    return w


def graph_ex2_1(xData, yData, yLabel, yNames, w, title=None):
    """
    Función auxiliar para dibujar los datos del ejercicio 2.1

    """
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
    """
    Obtiene el vector de datos etiquetados a partir de x.

    Parameters
    ----------
    x : Array
        Vector de características de x

    Returns
    -------
    Array
        Etiquetas para x

    """
    y = []
    for i in x:
        y.append(f(i[0], i[1]))
    
    return np.asarray(y)

def addNoise(y, percent):
    """
    Añadir ruido a las etiquetas.

    Parameters
    ----------
    y : Array
        Vector de etiquetas
    percent : Float
        Proporción de ruido entre [0, 1]

    Returns
    -------
    yNoided : Array
        Vector de etiquetas con ruido añadido.

    """
    
    # Por seguridad copiar el vector original.
    yNoided = np.copy(y)
    
    # Obtener la proporción por etiqueta del ruido, o sea dividir por 
    # la proporción y luego por la longitud de las etiquetas.
    noiseSize = int(len(y) * (percent / 2))

    # Obtener los índices de las etiquetas.
    posNoise = np.where(y == 1)
    negNoise = np.where(y == -1)

    # Barajear dichos índices in-place.
    np.random.shuffle(posNoise[0])
    np.random.shuffle(negNoise[0])
    
    # Obtener los primeros índices luego del barajeo hasta la proporción
    # adecuada.
    posNoise = posNoise[0][:noiseSize]
    negNoise = negNoise[0][:noiseSize]
    
    # Cambiar dichos valores en las etiquetas y retornarlas.
    yNoided[posNoise] = -1
    yNoided[negNoise] = 1

    return yNoided

def addBias(x):
    """
    Añadir una columna de 1s al principio, lo que sería el bias o sesgo.

    Parameters
    ----------
    x : Array
        Matriz de características

    Returns
    -------
    xBias : Array
        Misma matriz con columna de 1 añadida al principio.

    """
    bias = np.ones((x.shape[0],1))
    xBias = np.hstack((bias, x))
    
    return xBias

def phi(x):
    """
    Añadir componentes no lineales a la matriz de características de x.

    Parameters
    ----------
    x : Array
        Matriz de características x.

    Returns
    -------
    xNonL : Array
        Misma matriz con características no lineales añadidas.

    """
    # Hacer una nueva matriz con las columnas nuevas y mismas filas que x.
    nonLinear = np.empty((x.shape[0], 3))
    
    # Realizar lo pedido para la no linearidad:
    # - x_1 * x_2
    nonLinear[:, 0] = x[:, 0] * x[:, 1]
    # - x_1 ^ 2
    nonLinear[:, 1] = np.square(x[:, 0])
    # - x_2 ^ 2
    nonLinear[:, 2] = np.square(x[:, 1])
    
    # Juntarlo todo en una sola matriz
    xNonL = np.hstack((x, nonLinear))
    
    return xNonL
    

def graph_ex2_2(xData, yData, yLabel, w, title=None, showFit=True, nonLinear=False):
    """
    Función auxiliar para dibujar los datos el ejercicio 2.2
    
    """
    plt.figure()
    
    # Haciendo una lista de los colores para el Scatter Plot.
    colors = ['k', 'y']

    # Dibujar un scatter plot de los datos
    for i, label in enumerate(yLabel):
        x = xData[ np.where(yData == label) ][:, 1]
        y = xData[ np.where(yData == label) ][:, 2]
        plt.scatter(x, y, c=colors[i], label=label)
    
    # Localizar la leyenda para que no estorbe.
    plt.legend(loc='upper left')
    
    # Si se quiere mostrar un ajuste, se indica, y luego se indica que clase de ajuste.
    if(showFit):
        x1 = np.linspace(-1, 1, len(yData))
        # Si es lineal, se dibuja la recta.
        if(not nonLinear):
            x2 = (-w[0] - w[1] * x1) / w[2]    
            plt.plot(x1, x2, 'b-')
        else:
        # Si es no lineal, se realiza un "contorno" de la función para poder dibujarla bien.
            x2 = np.linspace(-1, 1, len(yData))
            X1, X2 = np.meshgrid(x1, x2)
            ZW = w[0] + (w[1] * X1 )+ (w[2] * X2) + (w[3] * X1 * X2 )+ (w[4] * np.square(X1)) + (w[5] * np.square(X2))
            
            plt.contour(X1, X2, ZW, 0, colors='b')
    
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.title(title)
    plt.show()

def acc(x, y, w):
    """
    Obtener el 'accuracy' del modelo.
    
    Calculado como Nº Aciertos / Nº Ejemplos Totales

    Parameters
    ----------
    x : Array
        Matriz de características de los datos.
    y : Array
        Vector de etiquetas de los datos.
    w : Array
        Pesos del modelo a evaluar.

    Returns
    -------
    perc : Float
        Porcentaje de aciertos

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

def hessianF(x,y):
    """
    Hessiana de la función f(x,y)

    """
    # Adaptado directamente de las segundas derivadas de la función.
    hess = np.array([[2 - 8 * np.square(np.pi) * np.sin(2 * np.pi * x) * np.sin(np.pi * y), 4 * np.square(np.pi) * np.cos(2 * np.pi * x) * np.cos(np.pi * y)],
                     [4 * np.square(np.pi) * np.cos(2 * np.pi * x) * np.cos(np.pi * y), 4 - 2 * np.square(np.pi) * np.sin(2 * np.pi * x) * np.sin(np.pi * y)]])
    
    return hess

def newton(w_ini, lr, grad_fun, fun, epsilon, max_iters):
    """
    Método de Newton para minimizar funciones
    
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
    wHist: Array
        Traza del valor de los pesos en cada iteración del algoritmo.
    errHist: Array
        Traza del valor del error de los pesos en cada iteración del algoritmo.

    """
    
    # Se inicializan los valores.
    errHist = []
    wHist= []
    
    # Idéntico a 'gradient_descent()'
    w = w_ini
    iterations = 0
    
    wHist.append(w)
    errHist.append(fun(w[0], w[1]))

    while(iterations < max_iters):
        
        # Ya que solo se probará la Hessiana de F, se llama directamente.
        hess = hessianF(w[0], w[1])
        
        # Se adapta la expresión matemática: Se invierte la Hessiana y se
        # multiplica por el gradiente; luego se multiplica eso por eta y
        # se le resta a los pesos.
        w = w - lr * np.dot(np.linalg.inv(hess), grad_fun(w[0], w[1]).T)
        
        # Almacenar el error y los pesos.
        errHist.append(fun(w[0], w[1]))
        wHist.append(w)

        iterations += 1 
        
        if(epsilon > fun(w[0], w[1])):
            break
        
    return w, iterations, wHist, errHist


def graph_bonus_1(it, hGD1, hGD2, hNewt1, hNewt2, title=""):
    """
    Función auxiliar para graficar el Bonus

    """
    
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
eta1 = 0.1 

# - Máximo número de iteraciones, el que venía en plantilla.
maxIter1 = 10000000000

# - El valor al que queremos llegar, sería el "error".
error1 = 1e-8

# - Punto inicial que se indica.
iniPoint1 = np.array([0.5,-0.5])

# Llamando a la función, pasándole el punto inicial, la tasa de aprendizaje,
# la función del gradiente, la función de error (aunque en este caso es la 
# función original), el error a obtener y el máximo de iteraciones.
w1, it1, _, wHist1 = gradient_descent(iniPoint1, eta1, gradE, E, error1, maxIter1)

#       1.2.b y 1.2.c:
print ('- Numero de iteraciones: ', it1)
print ('- Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')
print ("- Valor E(u,v) en punto:", E(w1[0], w1[1]))

display_figure(2, E, wHist1, 'coolwarm',"Evolución de los valores $w$ en $E(u,v)$", azimuth=100)

# 1.3:

#%% EJERCICIO 1.3.a
###################

print('Apartado 1.3.a')

# Por consistencia se redefinen variables para cada llamada del gradiente
# descendente.

iniPoint2 = [-1, 1]
eta2 = 0.01
# - Para evitar que el algoritmo se detenga antes de las 50 iteraciones, se fija
# el error a un valor inalcanzable.
error2 = -999
maxIter2 = 50

# Ejecudanto GD son eta=0.01
w2, it, errHistGD1, wHistGD1 = gradient_descent(iniPoint2, eta2, gradF, F, error2, maxIter2)

print("Con eta = 0.01")
print ('- Coordenadas obtenidas: (', w2[0], ', ', w2[1],')')
print ("- Valor f(x,y) en punto:", F(w2[0], w2[1]))

# Ejecudanto GD son eta=0.1
eta3 = 0.1
w3, it, errHistGD2, wHistGD2 = gradient_descent(iniPoint2, eta3, gradF, F, error2, maxIter2)

print("Con eta = 0.1")
print ('- Coordenadas obtenidas: (', w3[0], ', ', w3[1],')')
print ("- Valor f(x,y) en punto:", F(w3[0], w3[1]))

print("Imprimiendo gráficas...")
graph_ex1_3_a(it, errHistGD1, errHistGD2, ["$\eta=0.01$", "$\eta=0.1$"], title="Evolución de $f(x,y)$ por iteración dependiendo de $\eta$")

display_figure(2, F, wHistGD1, 'coolwarm',"Evolución del Gradiente de f(x,y), $\eta=0.01$", elev=35, azimuth=230)
display_figure(2, F, wHistGD2, 'coolwarm',"Evolución del Gradiente de f(x,y), $\eta=0.1$", elev=35, azimuth=230)

print("Listo.")

#%%
#   1.3.b
print('Apartado 1.3.b\n')

# Ahora se meten los valores en unas listas para hacer un bucle que
# calcule e imprima cada valor.
iniPointArr = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
etaArr = [0.01, 0.1]

wHist3 = []
funcHist3 = []

for i in iniPointArr:
    for j in etaArr:
        w, it, fH, wH = gradient_descent(i, j, gradF, F, error2, maxIter2)
        wHist3.append(wH)
        funcHist3.append(fH)

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
x_train1, y_train1 = readData('datos/X_train.npy', 'datos/y_train.npy')

# Lectura de los datos para el test
x_test1, y_test1 = readData('datos/X_test.npy', 'datos/y_test.npy')

# Se obtienen los pesos con la pseudoinversa en un solo cálculo.
wPseudo = pseudoinverse(x_train1, y_train1)

print ('Bondad del resultado para pseudoinversa:\n')
print (" - Pesos obtenidos: ", wPseudo)
print (" - E_in: ", Err(x_train1, y_train1, wPseudo), ", Acc: ", acc(x_train1, y_train1, wPseudo))
print (" - E_out: ", Err(x_test1, y_test1, wPseudo), ", Acc:", acc(x_test1, y_test1, wPseudo))

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
error = 0

# - Por defecto, se coloca un tamaño de batch a 32.
batchSize = 32

# Se realiza el algoritmo de SGD con los parámetros establecidos.
wSGD, iters = sgd(x_train1, y_train1, wIni, eta, batchSize, error, maxIters)

print ('\nBondad del resultado para grad. descendente estocástico:\n')
print (" - Pesos obtenidos: ", wSGD)
print (" - E_in: ", Err(x_train1, y_train1, wSGD), ", Acc: ", acc(x_train1, y_train1, wSGD))
print (" - E_out: ", Err(x_test1, y_test1, wSGD), ", Acc:", acc(x_test1, y_test1, wSGD))

# Se imprime la gráfica con los resultados.

graph_ex2_1(x_train1, y_train1, [-1, 1], [1, 5], wPseudo, title="Train: Pseudoinversa")
graph_ex2_1(x_train1, y_train1, [-1, 1], [1, 5], wSGD, title="Train: SGD")

graph_ex2_1(x_test1, y_test1, [-1, 1], [1, 5], wPseudo, title="Test: Pseudoinversa")
graph_ex2_1(x_test1, y_test1, [-1, 1], [1, 5], wSGD, title="Test: SGD")


#%% EJERCICIO 2.2
#################

print('Apartado 2.2.a-c\n')

# Defindiendo los vectores iniciales de pesos para los casos lineales y no
# lineales:
    
wIniLin = np.array([0, 0, 0])
wIniNonL = np.array([0, 0, 0, 0, 0, 0])

# Redefiniendo los parámetros de SGD:
maxIters = 200
wIni = np.array([0, 0, 0])
eta = 0.01
error = 0
batchSize = 32

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
x2Lin = addBias(x2)

#   - Usando como vector de características [1, x1, x2, x1x2, x1^2, x2^2]
x2NonL = phi(x2)
x2NonL = addBias(x2NonL)

#   -  Estimar el error de ajuste con características lineales.
wEndLin, _ = sgd(x2Lin, y2, wIniLin, eta, batchSize, error, maxIters)
#   - Ídem, pero no lineales.
wEndNonL, _ = sgd(x2NonL, y2, wIniNonL, eta, batchSize, error, maxIters)

print ('\nBondad del resultado para SGD con características lineales:\n')
print (" - Pesos obtenidos:", wEndLin)
print (" - E_in: ", Err(x2Lin, y2, wEndLin), ", Acc: ", acc(x2Lin, y2, wEndLin))

print ('\nBondad del resultado para SGD con características no lineales:\n')
print (" - Pesos obtenidos:", wEndNonL)
print (" - E_in: ", Err(x2NonL, y2, wEndNonL), ", Acc: ", acc(x2NonL, y2, wEndNonL))

#   - Graficar los datos, el ajuste con características lineales y con no lineales.
graph_ex2_2(x2Lin, y2, [-1, 1], wEndLin, title="Datos", showFit=False)
graph_ex2_2(x2Lin, y2, [-1, 1], wEndLin, title="Características Lineales")
graph_ex2_2(x2NonL, y2, [-1, 1], wEndNonL, title="Características No Lineales", nonLinear=True)

#%% EJERCICIO 2.2.d
###################
#   2.2.d:
#   - Ejecutar todo el experimento definido por (a)-(c) 1000 veces 
#   (generando 1000 muestras diferentes)

print("Apartado 2.2.d")

# Inicializando las variables pertinentes.
meanErrTrainLin = 0
meanErrTestLin = 0

meanAccTrainLin = 0
meanAccTestLin = 0

meanErrTrainNonL = 0
meanErrTestNonL = 0

meanAccTrainNonL = 0
meanAccTestNonL = 0

print("Realizando el experimento 1000 veces, ejecutando...")
print("0|          |100%\n |", end='')
for i in range(1000):
    
    # - Generando una muestra de N=1000 puntos en el cuadrado X=[-1, 1]x[-1, 1]
    # - Ahora test también.
    x3Train = simula_unif(1000, 2, 1)
    x3Test  = simula_unif(1000, 2, 1)
    
    # Asigando etiquetas a los puntos.
    y3Train = addNoise(getTags(x3Train), 0.1)
    y3Test  = addNoise(getTags(x3Test), 0.1)
    
    # Características:
    #   - Usando como vector de características [1, x1, x2], se deben de añadir
    #   tanto a test como train para poder evaluarlos luego.
    x3TrainLin = addBias(x3Train)    
    x3TestLin  = addBias(x3Test)
 
    #   - Usando como vector de características [1, x1, x2, x1x2, x1^2, x2^2]
    x3TrainNonL = phi(x3Train)
    x3TrainNonL = addBias(x3TrainNonL)
    
    x3TestNonL = phi(x3Test)
    x3TestNonL = addBias(x3TestNonL)
    
    #   -  Estimar el error de ajuste con características lineales.
    wEndLin, _ = sgd(x3TrainLin, y3Train, wIniLin, eta, batchSize, error, maxIters)
    #   - Ídem, pero no lineales.
    wEndNonL, _ = sgd(x3TrainNonL, y3Train, wIniNonL, eta, batchSize, error, maxIters)
    
    # Obteniendo el error y acc en train y test para características lineales:
    # - Error
    meanErrTrainLin += Err(x3TrainLin, y3Train, wEndLin)
    meanErrTestLin += Err(x3TestLin, y3Test, wEndLin)
    # - Accuracy
    meanAccTrainLin += acc(x3TrainLin, y3Train, wEndLin)
    meanAccTestLin +=  acc(x3TestLin, y3Test, wEndLin)
    
    # Obteniendo el error y acc en train y test para características no lineales:
    # - Error
    meanErrTrainNonL += Err(x3TrainNonL, y3Train, wEndNonL)
    meanErrTestNonL += Err(x3TestNonL, y3Test, wEndNonL)
    # - Accuracy
    meanAccTrainNonL += acc(x3TrainNonL, y3Train, wEndNonL)
    meanAccTestNonL +=  acc(x3TestNonL, y3Test, wEndNonL)
    
    if(i%100 == 0):
        print("█", end='')

print("|\n")

# Obteniendo la media de los datos pertinentes.
meanErrTrainLin = meanErrTrainLin / 1000
meanErrTestLin = meanErrTestLin / 1000

meanAccTrainLin = meanAccTrainLin / 1000
meanAccTestLin = meanAccTestLin / 1000

meanErrTrainNonL = meanErrTrainNonL / 1000
meanErrTestNonL = meanErrTestNonL / 1000

meanAccTrainNonL = meanAccTrainNonL / 1000
meanAccTestNonL = meanAccTestNonL / 1000

print("Resultados:")
print(" - Caracterísicas Lineales: ")
print("     > Ein medio:", meanErrTrainLin, ", Acc: ", meanAccTrainLin)
print("     > Eout medio:", meanErrTestLin, ", Acc:", meanAccTestLin)
print("\n - Caracterísicas No Lineales: ")
print("     > Ein medio:", meanErrTrainNonL, ", Acc: ", meanAccTrainNonL)
print("     > Eout medio:", meanErrTestNonL, ", Acc:", meanAccTestNonL)
#   2.2.e:
#   - En memoria.
    
#%% BONUS
#########

# Requiere la ejecución anterior de la celda 'EJERCICIO 1.3.a' 

print("\nBonus: Método de Newton")

# Repitiendo el experimento con los mismos parámetros que en 1.3.a

iniPoint = [-1, 1]
eta = 0.01
error = -999
maxIter = 50

wNewt1, it, wHistNewt1, errNewt1 = newton(iniPoint, eta, gradF, F, error, maxIter)
print("Probando a cambiar la tasa de aprendizaje...\n")

print("Con eta = 0.01")
print ('- Coordenadas obtenidas: (', wNewt1[0], ', ', wNewt1[1],')')
print ("- Valor f(x,y) en punto:", F(wNewt1[0], wNewt1[1]))

eta = 0.1
wNewt2, it, wHistNewt2, errNewt2 = newton(iniPoint, eta, gradF, F, error, maxIter)

print("Con eta = 0.1")
print ('- Coordenadas obtenidas: (', wNewt2[0], ', ', wNewt2[1],')')
print ("- Valor f(x,y) en punto:", F(wNewt2[0], wNewt2[1]))

print("Imprimiendo gráfica...")
graph_bonus_1(it, errHistGD1, errHistGD2, errNewt1, errNewt2, title="GD vs Newton: Error por iteración")
print("Listo.")

print("Probando a cambiar los puntos de inicio...\n")
startPoint = [[-0.5, -0.5], [1, 1], [2.1, -2.1], [-3, 3], [-2, 2]]
learningRates = [0.01, 0.1]

hist = []

for i in startPoint:
    for j in learningRates:
        w, it, _, h  = newton(i, j, gradF, F, error, maxIter)
        hist.append(h)
        print("Punto Inicial: ", i)
        print("Tasa Aprendizaje:",j)
        print("\t(x,y): ", w)
        print("\tf(x,y): ", F(w[0], w[1]))
        print("--------------------------")


for i in range(0, len(startPoint)):
    graph_bonus_1(it, funcHist3[i*2], funcHist3[(i*2)+1], hist[i*2], hist[(i*2)+1], title="GD vs Newton: Punto inicial "+ str(startPoint[i]))

