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


## FUNCIONES AUXILIARES

# FUNCIONES PARA EJERCICIOS

## FUNCIONES EJER 1

def scatterPlot(xData, yData=None, yLabel=None, line=None, N=None, colors=None, title=None):
    """
    WIP
    
    """
    plt.figure()
    
    # Haciendo una lista de los colores para el Scatter Plot.
    if yData is not None:
        
        actColors = cm.rainbow(np.linspace(0, 1, len(yLabel)))
        
        for i, label in enumerate(yLabel):
            x = xData[ np.where(yData == label) ][:, 0]
            y = xData[ np.where(yData == label) ][:, 1]    
            plt.scatter(x, y, color=actColors[i], label=str(label))
            
        plt.legend(loc='upper left')
        
        if line is not None:
            a = line[0]
            b = line[1]
            
            xLine = np.linspace(-N, N, len(yData))
            yLine = a * xLine + b
              
            plt.plot(xLine, yLine, 'c--')
            plt.xlim([-(N+1), N+1])
            plt.ylim([-(N+1), N+1])
        
    else:
        x = xData[:, 0]
        y = xData[:, 1]
        plt.scatter(x, y)
    
    
    # # Localizar la leyenda para que no estorbe.
    # plt.legend(loc='upper left')
        
    plt.title(title)
    plt.show()

## FUNCIONES EJER 1 FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1.1
#################

# Dibujar una gráfica con la nube de puntos de salida correspondiente.

x = simula_unif(50, 2, [-50,50])
scatterPlot(x, title="Distribución Uniforme: 50 puntos")

x = simula_gauss(50, 2, np.array([5,7]))
scatterPlot(x, title="Distribución Gaussiana: 50 puntos")

# input("\n--- Pulsar tecla para continuar ---\n")


#%% EJERCICIO 1.2 
#################

# Dibujar una gráfica con la nube de puntos de salida correspondiente

def signo(x):
# La funcion np.sign(0) da 0, lo que nos puede dar problemas
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

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


#%% 1.2.a: 
    
x = simula_unif(100, 2, [-50, 50])

y = getTags(x, 3, 1)

scatterPlot(x, yData=y, yLabel=[-1,1], line=[3, 1], N=50, title="Distribución Uniforme: 50 puntos")


# input("\n--- Pulsar tecla para continuar ---\n")

#%% 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    
    return ?  

#CODIGO DEL ESTUDIANTE

# Random initializations
iterations = []
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL(?):
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")
    


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos


# Funcion para leer los datos
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

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
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

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION 

#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")



#POCKET ALGORITHM
  
#CODIGO DEL ESTUDIANTE




input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

#CODIGO DEL ESTUDIANTE
