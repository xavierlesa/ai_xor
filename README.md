# Demo de una Red Neuronal (Neural Net) simulando un XOR

Este demo lo saque desde [aca](https://github.com/stmorgan/pythonNNexample/blob/master/PythonNNExampleFromSirajology.py)


## Principios

Basicamente en ML (sigla de Machine Learning) hay 3 partes:

1- El Modelado
2- El Entrenamiento
3- La Prueba


### Modelado

Hay mucho tipos de modelos de ML, pero el que se ha popularizado es el Neural Net
y tiende a asemejarse a una Nuerona y se basa en *layers* (capas).

Inputs, Hidden (computo?) y Outputs

![neural network](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)


### Entrenamiento

Le tiramos data con el *input* y el *output* para que *aprenda* a hacer el 
cálculo de nuestra función **XOR** `(!a.b + a.!b)`

Creamos un `array` de test con los datos de entrada y salida esperados y dejamos
que haga su magia.


 A | B | X | Q
---|---|---|---
 0 | 0 | 1 | 0
 0 | 1 | 1 | 1
 1 | 0 | 1 | 1
 1 | 1 | 1 | 0


A y B son los Inputs
[X] el bias es una especie de *activador* (aunque realmente no termino de entender que hace)
Q el resultado esperado

En el código es lo que creamos así:


```python
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
```


Por último para probar el código:

```bash
$ python demo.py
Error: 0.496410031903
Error: 0.00858452565325
Error: 0.00578945986251
Error: 0.00462917677677
Error: 0.00395876528027
Error: 0.00351012256786
Error: 0.00318350238587
Error: 0.00293230634228
Error: 0.00273150641821
Error: 0.00256631724004
Output pos entrenamiento
[[ 0.00199094]
 [ 0.99751458] # <--+ Mira mama muy cercano a 1 !
 [ 0.99771098] # <-/
 [ 0.00294418]]
```

[X]: http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
