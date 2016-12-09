#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np


def sigmoid(x, deriv=False):
    """
    Sigmod retorna un valor entre 0 y 1, es usado para genera la probabilidad 
    de un numero.
    
    https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide
    """
    if deriv:
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))


# input data A, B, X
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
    ])

# output data Q
y = np.array([
    [0],
    [1],
    [1],
    [0],
    ])

# setea el generador de num. aleatorios con una semilla conocida = 1
np.random.seed(1)

# Matriz de Sinapsis, es la conección entre una neurona y una capa y todas las neuronas
# de la capa siguinte. https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg
# Esta red es de 3 layers asi que solo tenemos 2 sinapsis.
#
# capa1 <-- sinapsis0 --> capa2 <-- sinapsis1 --> capa3
n_layers = 3
n_connections = 4
n_outputs = 1

# Matríz de sinapsis 0, conecta el layer 0 con el layer 1
# A [n,n,n,n,] 
# B [n,n,n,n,] 
# X [n,n,n,n,] 
syn0 = 2 * np.random.random((n_layers, n_connections)) - 1
 
# Matríz de sinapsis 1, conecta el layer 1 con el layer 2
#  Q
# [n]
# [n]
# [n]
# [n]
syn1 = 2 * np.random.random((n_connections, n_outputs)) - 1

# Etapa de entrenamiento sobre 60000 iteraciones y cada 10000 mostramos la 
# reducción del error.
n_trainning = 100000
n_error_debug = 10000
for j in xrange(n_trainning):

    layer0 = X
    layer1 = sigmoid(np.dot(layer0, syn0))
    layer2 = sigmoid(np.dot(layer1, syn1))

    layer2_err = y - layer2

    if (j % n_error_debug) == 0:
        print "Error: " + str(np.mean(np.abs(layer2_err)))


    layer2_delta = layer2_err * sigmoid(layer2, deriv=True)
    
    layer1_err = layer2_delta.dot(syn1.T)

    layer1_delta = layer1_err * sigmoid(layer1, deriv=True)

    # actualiza el weight (peso), algoritmo gradient descent
    syn1 += layer1.T.dot(layer2_delta)
    syn0 += layer0.T.dot(layer1_delta)

# Finalmente mostramos el resultado
print "Output pos entrenamiento"
print layer2
