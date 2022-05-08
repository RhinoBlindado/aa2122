#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[CASTELLANO]
 
    Práctica 2: Complejidad H y Modelos Lineales
    Asignatura: Aprendizaje Automático
    Autor: Valentino Lugli (Github: @RhinoBlindado)
    Abril 2022
    
[ENGLISH]

    Practice 2: H Complexity and Linear Models
    Course: Machine Learning
    Author: Valentino Lugli (Github: @RhinoBlindado)
    April 2022
"""

# LIBRERIAS

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Fijamos la semilla
np.random.seed(1)

# FUNCIONES AUXILIARES Y PROVISTAS

## FUNCIONES PROVISTAS

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gauss(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

def signo(x):
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    
    pred_y = []
    for i, j, _ in grid:
        pred_y.append(signo(fz(i, j)))
    
    pred_y = np.asarray(pred_y)
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    # ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                # cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    pos = []
    for i,j in positions.T:
        pos.append(fz(i,j))
    pos = np.asarray(pos)
    # ax.contour(XX,YY,pos.reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()

## FUNCIONES AUXILIARES

def stop():
    """
    Detiene la ejecución hasta que se presione 'Enter'

    """
    input("Presiona Enter para continuar. . .")
    pass

def progBarHeader(step):
    """
    Dibuja una barra de carga que aumenta cada <step>%

    """
    print("0|",end='')
    for i in range(0, 100, step):
        print(" ", end='')

    print("|100%\n |", end='')

def progBarFooter():
    print("|\n")

def progBar(actual, total, step, nextStep, char='█'):
    
    ratio = actual / total
        
    if((ratio * 100) >= nextStep):
        print(char, end='')
        nextStep += step
        
    return nextStep


# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 1

def scatterPlot(xData, yData=None, yLabel=None, yNames=None, line=None, plotRange=None, colors=None, title=None):
    """
    WIP
    
    """
    plt.figure()
    
    if yData is not None:
        
        actColors = cm.rainbow(np.linspace(0, 1, len(yLabel)))
        
        for i, label in enumerate(yLabel):
            x = xData[ np.where(yData == label) ][:, 0]
            y = xData[ np.where(yData == label) ][:, 1]    
            if yNames is None:
                plt.scatter(x, y, color=actColors[i], label="Etiqueta {}".format(str(label)))
            else:
                plt.scatter(x, y, color=actColors[i], label="{}".format(str(yNames[i])))

        
        if line is not None:
            loB = plotRange[0]
            hiB = plotRange[1]
            
            a = line[0]
            b = line[1]
            
            xLine = np.linspace(loB, hiB, len(yData))
            if len(line) == 2:
                yLine = a * xLine + b
                                  
            elif len(line) == 3:
                c = line[2]
                yLine = (-a - b * xLine) / c
                  
            plt.plot(xLine, yLine, 'c--', label="Recta")
            
            if len(plotRange) == 2:
                plt.xlim([loB, hiB])
                plt.ylim([loB, hiB])
                
            if len(plotRange) == 4:
                plt.xlim([plotRange[0],  plotRange[1]])
                plt.ylim([plotRange[2],  plotRange[3]])
                
        plt.legend(loc='upper right')

    else:
        x = xData[:, 0]
        y = xData[:, 1]
        plt.scatter(x, y)    
    
    plt.title(title)
    plt.show()


def scatterPlotNonL(xData, yData=None, yLabel=None, yNames=None, function=None, plotRange=None, title=None):
    """
    WIP
    
    """
    plt.figure()
    
    if yData is not None:
        
        if function is not None:
            loB = plotRange[0]
            hiB = plotRange[1]
            
            xAx = np.linspace(loB, hiB, len(yData))
            yAx = xAx
            
            X, Y = np.meshgrid(xAx, yAx)
            
            Z = function(X, Y)
            
            plt.contour(X, Y, Z, 0, colors='c', linestyles='dashed')
            # Z = np.clip(Z, -1, 1)
            plt.contourf(X, Y, Z, 50, cmap='RdBu',vmin=-1, vmax=1)
            # clrBar = plt.colorbar(clrs)
            # clrBar.set_label('Signo')
            # clrBar.set_ticks([-1, 0, 1])

            if len(plotRange) == 2:
                plt.xlim([loB, hiB])
                plt.ylim([loB, hiB])
                
            if len(plotRange) == 4:
                plt.xlim([plotRange[0],  plotRange[1]])
                plt.ylim([plotRange[2],  plotRange[3]])
        
        actColors = cm.rainbow(np.linspace(0, 1, len(yLabel)))
        
        for i, label in enumerate(yLabel):
            x = xData[ np.where(yData == label) ][:, 0]
            y = xData[ np.where(yData == label) ][:, 1]    
            if yNames is None:
                plt.scatter(x, y, color=actColors[i], label="Etiqueta {}".format(str(label)))
            else:
                plt.scatter(x, y, color=actColors[i], label="{}".format(str(yNames[i])))


        plt.legend(loc='upper right')

    else:
        x = xData[:, 0]
        y = xData[:, 1]
        plt.scatter(x, y)
    
    actTextColr = cm.RdBu(np.linspace(0, 1, 2))

    plt.text(-15, -65, "Zona -", alpha=0.75, color="black", backgroundcolor="white")
    plt.text(-3, -65, "█", color=actTextColr[0], backgroundcolor="white")
    plt.text(3, -65, "Zona +", color="black", backgroundcolor="white")
    plt.text(15, -65, "█", color=actTextColr[1], backgroundcolor="white")

    plt.title(title)
    plt.show()


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

    # Obtener los índices de las etiquetas.
    posNoise = np.where(y == 1)
    negNoise = np.where(y == -1)

    # Barajear dichos índices in-place.
    np.random.shuffle(posNoise[0])
    np.random.shuffle(negNoise[0])
    
    # Obtener los primeros índices luego del barajeo hasta la proporción
    # adecuada.
    
    noiseSizePos = int(np.round(len(posNoise[0]) * percent))
    noiseSizeNeg = int(np.round(len(negNoise[0]) * percent))
        
    posNoise = posNoise[0][:noiseSizePos]
    negNoise = negNoise[0][:noiseSizeNeg]
    
    # Cambiar dichos valores en las etiquetas y retornarlas.
    yNoided[posNoise] = -1
    yNoided[negNoise] = 1

    return yNoided


def getTags(x, a, b):
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
        y.append(f(i[0], i[1], a, b))
    
    return np.asarray(y)

    
def frontierFun1(x, y):
    return (np.square(x - 10) + np.square(y - 20) - 400)

def frontierFun2(x, y):
	return (0.5 * np.square(x + 10) + np.square(y - 20) - 400)

def frontierFun3(x, y):
	return (0.5 * np.square(x - 10) - np.square(y + 20) - 400)

def frontierFun4(x, y):
	return (y - 20 * np.square(x) - 5 * x + 3)

def getTagsFF(x, fun):
    y = []
    for i in x:
        y.append(signo(fun(i[0], i[1])))
    
    return np.asarray(y) 

def getFunAcc(x, y, fun):
    
    hit = 0
    
    for x_i, y_i in zip(x,y):
        if(signo(fun(x_i[0], x_i[1])) == y_i):
            hit += 1
    
    return hit / len(y)

## FUNCIONES EJER 1 FIN

## FUNCIONES EJER 2

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

def ajusta_PLA(datos, label, max_iter, vini, pocket=False):
    """
    Implementación del Algoritmo Perceptrón (PLA) y Perceptrón-Pocket (PLA-Pocket)

    Parameters
    ----------
    datos : Array
        Vector de características.
    label : Array
        Vector de etiquetas.
    max_iter : Int
        Número máximo de iteraciones.
    vini : Array
        Pesos iniciales.
    pocket : Boolean, opcional
        Modo Pocket. Por defecto False, modo PLA.

    Returns
    -------
    w : Array
        Pesos finales obtenidos en el caso de PLA.
        Mejores pesos obtenidos en el caso de PLA-Pocket
    i : Int
        Iteraciones necesarias para obtener dichos pesos.

    """
    
    # Inicialización básica
    i = 0
    w = vini
    
    # Para la versión Pocket:
    # - El mejor peso obtenido
    bestW = None
    # - El mejor error obtenido, inicializado a "+infinito".
    bestErr = float('inf')
        
    # Mientras no se haya llegado al máximo de iteraciones...
    while (i < max_iter):
        
        # Variable bandera que permite determinar si en alguna iteracion se 
        # modifican lo pesos.
        modified = False
        
        # ... y para cada x en datos y su correspondiente y en label...
        for x, y in zip(datos, label):
            
            # ... Si el signo del producto de x con w no es igual que y...
            if(signo(x.dot(w)) != y):
                # ... modificar el peso para que se ajuste a los datos..
                w = w + x*y
                # ... activar la bandera de modificación.
                modified = True
            
            # ... Si es la versión Pocket ...
            if(pocket):
                # ... obtener el error actual ...
                actErr = errPLA(datos, label, w)
                
                # ...si es mejor que el mejor error, 
                # actualizar y guardar el peso.
                if (actErr < bestErr):
                    bestW = w
                    bestErr = actErr
        
        # ... Si ha habido un pase entero por los datos y no se modificó w,
        # entonces parar el bucle.
        if (not modified):
            break
        
        i += 1

    # Si es la versión pocket, se devuelve el mejor peso en vez del último.
    if (pocket):
        w = bestW
        
    # Devolver el último peso, si es PLA o el mejor peso si es PLA-Pocket 
    # junto con las iteraciones.
    return w, i

def accPLA(x, y, w):
    """
    Obtener el accuracy de los pesos ajustados por PLA

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


def errPLA(x, y, w):
    h_w = x.dot(w)
    
    diffCount = 0
    
    for wTxi, yi in zip(h_w, y):
        
        if(signo(wTxi) != signo(yi)):
            diffCount += 1
    
    return diffCount / len(y)

def accRL(x, y, w):
    
    h_w = x.dot(w)
    probs = 1 / (1 + np.exp(-h_w))
    
    probs[ probs >= 0.5 ] = 1
    probs[ probs <  0.5 ] = -1
    
    good = 0
    
    for yHat, yReal in zip(probs, y):
        if(signo(yHat) == signo(yReal)):
            good += 1

    return good / len(y)


def errRL(x, y, w):
    
    h_w = x.dot(w)
    return np.mean( np.log( 1 + np.exp( -y * h_w ) ) )

def gradRL(x, y, w):
    """
    
    
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
        
    h_w = x.dot(w)
    yTemp = np.reshape(y, ([y.shape[0], 1]))
    
    numerator = yTemp * x
    denominator = 1 + np.exp( y * h_w )
    
    return -np.mean( numerator.T / denominator, axis=1)

def sgd(x, y, wIni, lr, batchSize, maxIters, gradFun, wStop = True):
    """
    WIP

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
        wOld = w
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
            w = w - lr * gradFun(xShuff[ini:fin], yShuff[ini:fin], w)
            
            # Sumar una iteración.
            i += 1 
        
            # Si se obtiene el máximo de iteraciones dentro de este bucle, 
            # cortar la ejecución también.
            if(i == maxIters): break
            
        # Si el error después de una época es menor al que se desea, también
        # finalizar.              
        if(wStop and np.linalg.norm( np.abs(wOld - w) ) < 0.01):
            break
        
    return w, i

def getBestParams(x, y, wIn, eta, batch, maxIters, gradFun):
    
    wIni = wIn
    
    etas = np.arange(eta[0], eta[1]+eta[2], eta[2])
    batches = np.arange(batch[0], batch[1]+batch[2], batch[2])
    
    bestScore = float('inf')
    
    bestW = None
    bestEta = None
    bestBatch = None
    bestIt = None
    
    step = 2
    nxtStep = 0
    actual = 0
    total = len(etas) * len(batches)
    
    progBarHeader(step)
    
    for i in batches:
        for j in etas:
            wEnd, it = sgd(x, y, wIni, j, i, maxIters, gradFun)
            
            score = errRL(x, y, wEnd)
            
            actual += 1
            if(score < bestScore):
                # print("BEST")
                # print(i,j, score, it)
                bestScore = score
                
                bestW = wEnd
                bestEta = j
                bestBatch = i
                bestIt = it
                
            nxtStep = progBar(actual, total, step, nxtStep, "|")
            
            
    progBarFooter()
    return bestW, bestEta, bestBatch, bestIt
    
## FUNCIONES EJER 2 FIN

## FUNCIONES BONUS

def gradLinR(x, y, w):
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

def errLinR(x, y, w):
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
    

def evalFunction(x_train, y_train, x_test, y_test, w, accFun, errFun, algoName):
    
    scatterPlot(x_train[:,1:], yData=y_train, yLabel=[1, -1], yNames=["8", "4"], line=w, plotRange=[0, 1, -7.5, 0.5], title="{}: Entrenamiento".format(algoName))

    print("\t - Pesos:\t\t", w)
    print("\t - Error:\t\t", errFun(x_train, y_train, w))
    print("\t - Accuracy:\t {:.2f}%".format(accFun(x_train, y_train, w) * 100))

    print("\tTest:")

    scatterPlot(x_test[:,1:], yData=y_test, yLabel=[1, -1], yNames=["8", "4"], line=w, plotRange=[0, 1, -7.5, 0.5], title="{}: Test".format(algoName))

    print("\t - Error:\t\t", errFun(x_test, y_test, w))
    print("\t - Accuracy:\t {:.2f}%".format(accFun(x_test, y_test, w) * 100))

## FUNCIONES BONUS FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

### EJERCICIO 1
#############

#%% EJERCICIO 1.1
#############

# Dibujar una gráfica con la nube de puntos de salida correspondiente.

print("+-------------+")
print("|Ejercicio 1.1|")
print("+-------------+\n")

# Obteniendo los puntos con la distribución uniforme y dibujándolos.
x1_1 = simula_unif(50, 2, [-50,50])
scatterPlot(x1_1, title="Distribución Uniforme: 50 puntos")

# Obteniendo los puntos con la distribución gaussiana y dibujándolos.
x1_1 = simula_gauss(50, 2, [5,7])
scatterPlot(x1_1, title="Distribución Gaussiana: 50 puntos")

### EJERCICIO 1.2 
#############

#%% EJERCICIO 1.2.a
#############

print("+---------------+")
print("|Ejercicio 1.2.a|")
print("+---------------+\n")

x1_2 = simula_unif(100, 2, [-50, 50])

a, b = simula_recta([11, 16])

print("Generando recta con valores a = {:.4f} y b = {:.4f}".format(a, b))

y1_2 = getTags(x1_2, a, b)

scatterPlot(x1_2, yData=y1_2, yLabel=[1,-1], line=[a, b], plotRange=[-50,50], title="Etiquetado perfecto con recta $y={:.2f} + x \cdot {:.2f}$".format(b,a))


#%% EJERCICIO 1.2.b
#############

print("+---------------+")
print("|Ejercicio 1.2.b|")
print("+---------------+\n")

# Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

y1_2Noise = addNoise(y1_2, 0.1)

scatterPlot(x1_2, yData=y1_2Noise, yLabel=[1,-1], line=[a, b], plotRange=[-50,50], title="Etiquetado con 10% de error; recta $y={:.2f} + x \cdot {:.2f}$".format(b,a))

# stop()

#%% EJERCICIO 1.2.c
#############
# Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

print("+---------------+")
print("|Ejercicio 1.2.c|")
print("+---------------+\n")

fNames = ["f_1", "f_2", "f_3", "f_4"]
fFuncs = [frontierFun1, frontierFun2, frontierFun3, frontierFun4]

print("Funciones más complejas con etiquetado lineal:")


for name, func in zip(fNames, fFuncs):
    
    print(" - {} Accuracy:\t {:.2f}%".format(name, getFunAcc(x1_2, y1_2Noise, func) * 100))
    scatterPlotNonL(x1_2, yData=y1_2Noise, yLabel=[1, -1], function=func, plotRange=[-50, 50], title="Función ${}(x,y)$ con etiquetas lineales + ruido".format(name))

print("\nFunciones más complejas con etiquetado etiquetado propio + ruido:")

for name, func in zip(fNames, fFuncs):
    
    yf = getTagsFF(x1_2, func)
    yfN = addNoise(yf, 0.1)
    
    print(" - {} Accuracy:\t {:.2f}%".format(name, getFunAcc(x1_2, yfN, func) * 100))
    scatterPlotNonL(x1_2, yData=yfN, yLabel=[1, -1], function=func, plotRange=[-50, 50], title="Función ${}(x,y)$ con etiquetas propias + ruido".format(name))


# stop()

### EJERCICIO 2 
#############

### EJERCICIO 2.1 
#############

#%% EJERCICIO 2.1.a
#############
# ALGORITMO PERCEPTRON

print("+---------------+")
print("|Ejercicio 2.1.a|")
print("+---------------+\n")

print("PLA: Usando datos del ejercicio 1.2.a\n")

PLA_MAXITERS = 5000
wIni = np.array([0, 0, 0])
w, iters = ajusta_PLA(addBias(x1_2), y1_2, PLA_MAXITERS, wIni)

print("Entrenamiento con vector inicializado a cero:")
print(" - Pesos:\t\t", w)
print(" - Iteraciones:\t", iters)
print(" - Accuracy:\t {:.2f}%".format(accPLA(addBias(x1_2), y1_2, w) * 100))

scatterPlot(x1_2, yData=y1_2, yLabel=[1,-1], line=w, plotRange=[-50,50], title="PLA: Ajuste Perfecto, inicialización a cero.")

# Random initializations
wIniArr = []
wFinArr = []
itArr   = []

itMean  = 0

print("\nEntrenamiento con vectores inicializados a valores entre [0, 1]:")
print(" # | Pesos Iniciales                | Pesos Finales                     | Accuracy  | Iteraciones")
print("---+--------------------------------+-----------------------------------+-----------+------------")
for i in range(0,10):
    wIniTemp = np.random.uniform(0, 1, 3)
    wEndTemp, itTemp = ajusta_PLA(addBias(x1_2), y1_2, PLA_MAXITERS, wIniTemp)
    
    itMean += itTemp

    print("{:2d} | [{: 8.4f}, {: 8.4f}, {: 8.4f}] | [{: 8.4f}, {: 8.4f}, {: 8.4f}]    | {: 3.2f}%  | {:5d}" \
          .format(i, wIniTemp[0], wIniTemp[1], wIniTemp[2], wEndTemp[0], wEndTemp[1], wEndTemp[2], \
              (accPLA(addBias(x1_2), y1_2, wEndTemp) * 100), itTemp))

itMean /= 10
print('\nValor medio de iteraciones necesario para converger: {}'.format(itMean))

# stop()

#%% EJERCICIO 2.1.b
#############
# Ahora con los datos del ejercicio 1.2.b

print("+---------------+")
print("|Ejercicio 2.1.b|")
print("+---------------+\n")

print("PLA: Usando datos del ejercicio 1.2.b\n")

wIni = np.array([0, 0, 0])
w, iters = ajusta_PLA(addBias(x1_2), y1_2Noise, PLA_MAXITERS, wIni)

print("Entrenamiento con vector inicializado a cero:")
print(" - Pesos:\t\t", w)
print(" - Iteraciones:\t", iters)
print(" - Accuracy:\t {:.2f}%".format(accPLA(addBias(x1_2), y1_2Noise, w) * 100))

scatterPlot(x1_2, yData=y1_2Noise, yLabel=[1,-1], line=w, plotRange=[-50,50], title="PLA: 10% ruido, inicialización a cero.")

# Random initializations
wIniArr = []
wFinArr = []
itArr   = []

itMean  = 0
accMean = 0

print("\nEntrenamiento con vectores inicializados a valores entre [0, 1]:")
print(" # | Pesos Iniciales                | Pesos Finales                   | Accuracy | Iteraciones")
print("---+--------------------------------+---------------------------------+----------+------------")
for i in range(0,10):
    wIniTemp = np.random.uniform(0, 1, 3)
    wEndTemp, itTemp = ajusta_PLA(addBias(x1_2), y1_2Noise, PLA_MAXITERS, wIniTemp)
    
    itMean += itTemp
    actAcc = accPLA(addBias(x1_2), y1_2Noise, wEndTemp)
    accMean += actAcc
    print("{:2d} | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | {: 3.2f}%  | {:5d}" \
          .format(i, wIniTemp[0], wIniTemp[1], wIniTemp[2], wEndTemp[0], wEndTemp[1], wEndTemp[2], \
              (actAcc * 100), itTemp))

itMean /= 10
accMean /= 10
print('\nValor medio de accuracy: {: 3.2f}%'.format(accMean * 100))

# stop()

#%% EJERCICIO 2.2
#############
#REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

print("+-------------+")
print("|Ejercicio 2.2|")
print("+-------------+\n")


print("Obteniendo 100 datos de muestra...")
x2_train = simula_unif(100, 2, [0, 2])

a_2, b_2 = simula_recta([0, 2])

y2_train = getTags(x2_train, a_2, b_2)

scatterPlot(x2_train, yData=y2_train, yLabel=[1,-1], line=[a_2, b_2], plotRange=[-0.25,2.25], title="100 datos generados con recta ideal")

RL_MAXITERS = 100000
wIni = np.array([0, 0, 0])

x2_trainB = addBias(x2_train)

print("Entrenamiento con mejores parámetros:")
# etas = [0.0001, 0.1, 0.0005]
# batches = [2, 100, 2]
# bestW, lr, batchSize, it_2 = getBestParams(x2_trainB, y2_train, wIni, etas, batches, RL_MAXITERS, gradRL)

batchSize = 2
lr = 0.09960000000000001

bestW, it_2 = sgd(x2_trainB, y2_train, wIni, lr, batchSize, RL_MAXITERS, gradRL)

scatterPlot(x2_train, yData=y2_train, yLabel=[1,-1], line=bestW, plotRange=[-0.25, 2.25], title="RL: Recta obtenida por SGD para datos de entrenamiento")

print(" - Batch Size:\t", batchSize)
print(" - Eta:\t\t\t", lr)
print(" - Pesos:\t\t", bestW)
print(" - Error:\t\t", errRL(x2_trainB, y2_train, bestW))
print(" - Accuracy:\t{: 3.2f}%".format(accRL(x2_trainB, y2_train, bestW) * 100))
print(" - Iteraciones:\t", it_2)

# stop()

# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


x2_test = simula_unif(1611, 2, [0, 2])
y2_test = getTags(x2_test, a_2, b_2)

print("Resultado para conjunto de test con tamaño >999:")
print(" - Error:\t\t", errRL(addBias(x2_test), y2_test, bestW))
print(" - Accuracy:\t{: 3.2f}%".format(accRL(addBias(x2_test), y2_test, bestW) * 100))
scatterPlot(x2_test, yData=y2_test, yLabel=[1,-1], line=bestW, plotRange=[-0.25, 2.25], title="RL: Recta en nuevos datos generados de test")

# Comentar en memoria primer experimento con estos datos: 
# etas = [0.0001, 0.1, 0.0005]
# batches = [2, 100, 2]


totalErr = 0
totalIts = 0
totalAcc = 0

print("Repitiendo el experimento 100 veces:")

progBarHeader(5)
nxtStep = 5

for i in range(0, 100):

    # Obteniendo una recta por ese espacio
    ai, bi = simula_recta([0, 2])

    # Generando 100 datos de entrenamiento
    xi_train = simula_unif(100, 2, [0, 2])
    yi_train = getTags(xi_train, ai, bi)
    
    # Generando >999 datos de test
    xi_test = simula_unif(1611, 2, [0, 2])
    yi_test = getTags(xi_test, ai, bi)

    # Entrenando en los datos
    wi, iti = sgd(addBias(xi_train), yi_train, wIni, lr, batchSize, RL_MAXITERS, gradRL)
    
    # Acumulando las métricas pedidas
    totalErr += errRL(addBias(xi_test), yi_test, wi)
    totalIts += iti
    totalAcc += accRL(addBias(xi_test), yi_test, wi)
    
    # Imprimir la barra de carga cada vez que se progresa un 5%
    nxtStep = progBar(i, 100, 5, nxtStep)

progBar(100, 100, 5, nxtStep)
progBarFooter()
    
totalErr /= 100
totalIts /= 100
totalAcc /= 100

print("Resultados:")
print(" - E_out medio:\t\t\t", totalErr)
print(" - Accuracy medio:\t\t{: 3.2f}%".format(totalAcc * 100))
print(" - Iteraciones medias:\t", totalIts)


#%% BONUS
#########

print("+-----+")
print("|Bonus|")
print("+-----+\n")

#Clasificación de Dígitos

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()


#%%

print("+---------+")
print("|Bonus 2.a|")
print("+---------+\n")

print("Regresión Lineal: Pseudoinversa")
print("-------------------------------")

wPseudo = pseudoinverse(x, y)

evalFunction(x, y, x_test, y_test, wPseudo, accPLA, errLinR, "Pseudoinversa")


#%%

print("PLA")
print("---")
wPLA, _ = ajusta_PLA(x, y, PLA_MAXITERS, [0, 0 ,0])

evalFunction(x, y, x_test, y_test, wPLA, accPLA, errPLA, "PLA")


#%%
print("PLA-Pocket")
print("----------")

wPocket, _ = ajusta_PLA(x, y, PLA_MAXITERS, [0, 0 ,0], pocket=True)

evalFunction(x, y, x_test, y_test, wPocket, accPLA, errPLA, "PLA-Pocket")


#%%

print("Regresión Logística")
print("-------------------")

# batchSize = 2
# lr = 0.09960000000000001

#etas = [0.001, 0.1, 0.0005]
#batches = [2, 100, 2]
#bestW, lr, batchSize, it_2 = getBestParams(x, y, [0,0,0], etas, batches, RL_MAXITERS, gradRL)

batchSize = 2
lr=  0.08700000000000001

wRL, _ = sgd(x, y, [0, 0, 0], lr, batchSize, RL_MAXITERS, gradRL)

evalFunction(x, y, x_test, y_test, wRL, accRL, errRL, "RL")


#%% Bonus 1.c

print("+---------+")
print("|Bonus 2.c|")
print("+---------+\n")

wIniBonus = wPseudo

print("PLA con LinR")
print("------------")
wPLA, _ = ajusta_PLA(x, y, PLA_MAXITERS, wIniBonus)

evalFunction(x, y, x_test, y_test, wPLA, accPLA, errPLA, "PLA+LinR")

print("PLA-Pocket con LinR")
print("-------------------")

wPocket, _ = ajusta_PLA(x, y, PLA_MAXITERS, wIniBonus, pocket=True)

evalFunction(x, y, x_test, y_test, wPocket, accPLA, errPLA, "PLA-Pocket+LinR")

print("Regresión Logística con LinR")
print("----------------------------")

wRL, _ = sgd(x, y, wIniBonus, lr, batchSize, RL_MAXITERS, gradRL)

evalFunction(x, y, x_test, y_test, wRL, accRL, errRL, "RL+LinR")


#%%
print("+---------+")
print("|Bonus 2.d|")
print("+---------+\n")

delta = 0.05
dVC = 3
N_in = len(y)
N_test = len(y_test)

eIns = []


