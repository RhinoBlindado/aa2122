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

def stop():
    # input("Presiona cualquier tecla para continuar")
    pass

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
            plt.scatter(x, y, color=actColors[i], label="Etiqueta {}".format(str(label)))
            
        plt.legend(loc='upper right')
        
        if line is not None:
            if len(line) == 2:
                a = line[0]
                b = line[1]
                
                xLine = np.linspace(-N, N, len(yData))
                yLine = a * xLine + b
                  
                plt.xlim([-(N+1), N+1])
                plt.ylim([-(N+1), N+1])
                
            elif len(line) == 3:
                a = line[0]
                b = line[1]
                c = line[2]
                
                xLine = np.linspace(-N, N, len(yData))
                yLine = (-a - b * xLine) / c
                  
            plt.plot(xLine, yLine, 'c--', label="Recta")
            
            plt.xlim([-(N+1), N+1])
            plt.ylim([-(N+1), N+1])
            plt.legend(loc='upper left')

        
    else:
        x = xData[:, 0]
        y = xData[:, 1]
        plt.scatter(x, y)
    
    
    # # Localizar la leyenda para que no estorbe.
    # plt.legend(loc='upper left')
        
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
## FUNCIONES EJER 1 FIN

# FIN FUNCIONES PARA EJERCICIOS

# IMPLEMENTACION EJERCICIOS

#%% EJERCICIO 1.1
#################

# Dibujar una gráfica con la nube de puntos de salida correspondiente.

x1_1 = simula_unif(50, 2, [-50,50])
scatterPlot(x1_1, title="Distribución Uniforme: 50 puntos")

x1_1 = simula_gauss(50, 2, np.array([5,7]))
scatterPlot(x1_1, title="Distribución Gaussiana: 50 puntos")

stop()

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


#%% EJERCICIO 1.2.a
#############

x1_2 = simula_unif(100, 2, [-50, 50])

a, b = simula_recta([1, 3])

y1_2 = getTags(x1_2, a, b)

scatterPlot(x1_2, yData=y1_2, yLabel=[-1,1], line=[a, b], N=50, title="Ajuste perfecto; recta $y={:.2f} + x \cdot {:.2f}$".format(b,a))


stop()

#%% EJERCICIO 1.2.b
#############

# Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

y1_2Noise = addNoise(y1_2, 0.1)

scatterPlot(x1_2, yData=y1_2Noise, yLabel=[-1,1], line=[a, b], N=50, title="Ajuste con 10% de error; recta $y={:.2f} + x \cdot {:.2f}$".format(b,a))

stop()

#%% EJERCICIO 1.2.c
#############
# Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

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
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
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

#%%

yf1 = getTagsFF(x1_2, frontierFun1)
yf1Noise = addNoise(yf1, 0.1)

plot_datos_cuad(x, yf1Noise, frontierFun1, title='Point cloud plot', xaxis='x axis', yaxis='y axis')

# plot_datos_cuad(x, yNoise, frontierFun2, title='Point cloud plot', xaxis='x axis', yaxis='y axis')

# plot_datos_cuad(x, yNoise, frontierFun3, title='Point cloud plot', xaxis='x axis', yaxis='y axis')

# plot_datos_cuad(x, yNoise, frontierFun4, title='Point cloud plot', xaxis='x axis', yaxis='y axis')


stop()


#%% EJERCICIO 2.1.b 
# ALGORITMO PERCEPTRON

def ajusta_PLA(datos, label, max_iter, vini):
    
    i = 0
    w = vini
    
    while(i < max_iter):
        modified = False
        for x, y in zip(datos, label):
            
            if(signo(x.dot(w)) != y):
                w = w + x*y
                modified = True
        
        if(not modified):
            break
        
        i += 1
        
        
    return w, i  

def accPLA(x, y, w):
    
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

scatterPlot(x1_2, yData=y1_2, yLabel=[-1,1], line=w, N=50, title="PLA: Ajuste Perfecto, inicialización a cero.")

# Random initializations
wIniArr = []
wFinArr = []
itArr   = []

itMean  = 0

print("\nEntrenamiento con vectores inicializados a valores entre [0, 1]:")
print(" # | Pesos Iniciales                | Pesos Finales                   | Accuracy | Iteraciones")
print("---+--------------------------------+---------------------------------+----------+------------")
for i in range(0,10):
    wIniTemp = np.random.uniform(0, 1, 3)
    wEndTemp, itTemp = ajusta_PLA(addBias(x1_2), y1_2, PLA_MAXITERS, wIniTemp)
    
    itMean += itTemp

    print("{:2d} | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | {: 3.2f}%  | {:5d}" \
          .format(i, wIniTemp[0], wIniTemp[1], wIniTemp[2], wEndTemp[0], wEndTemp[1], wEndTemp[2], \
              (accPLA(addBias(x1_2), y1_2, wEndTemp) * 100), itTemp))

itMean /= 10
print('\nValor medio de iteraciones necesario para converger: {}'.format(itMean))

stop()

#%% EJERCICIO 2.1.b
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

scatterPlot(x1_2, yData=y1_2Noise, yLabel=[-1,1], line=w, N=50, title="PLA: 10% ruido, inicialización a cero.")

# Random initializations
wIniArr = []
wFinArr = []
itArr   = []

itMean  = 0

print("\nEntrenamiento con vectores inicializados a valores entre [0, 1]:")
print(" # | Pesos Iniciales                | Pesos Finales                   | Accuracy | Iteraciones")
print("---+--------------------------------+---------------------------------+----------+------------")
for i in range(0,10):
    wIniTemp = np.random.uniform(0, 1, 3)
    wEndTemp, itTemp = ajusta_PLA(addBias(x1_2), y1_2Noise, PLA_MAXITERS, wIniTemp)
    
    itMean += itTemp

    print("{:2d} | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | ({: 8.4f}, {: 8.4f}, {: 8.4f}) | {: 3.2f}%  | {:5d}" \
          .format(i, wIniTemp[0], wIniTemp[1], wIniTemp[2], wEndTemp[0], wEndTemp[1], wEndTemp[2], \
              (accPLA(addBias(x1_2), y1_2Noise, wEndTemp) * 100), itTemp))

itMean /= 10
print('\nValor medio de iteraciones necesario para converger: {}'.format(itMean))

stop()

###############################################################################
###############################################################################
###############################################################################

#%% EJERCICIO 2.2
#REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

print("+-------------+")
print("|Ejercicio 2.2|")
print("+-------------+\n")

# def 

def sgd(x, y, wIni, lr, batchSize, maxIters, gradFun):
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
            w = w - lr * gradFun(w, xShuff[ini:fin], yShuff[ini:fin])
            
            # Sumar una iteración.
            i += 1 
            
            # Si se obtiene el máximo de iteraciones dentro de este bucle, 
            # cortar la ejecución también.
            if(i < maxIters): break
            
        # Si el error después de una época es menor al que se desea, también
        # finalizar.
        # if(Err(x, y, w) < epsilon):
            # break
        
    return w, i

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
