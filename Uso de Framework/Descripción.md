# Portafolio de Implementación: Uso de framework de aprendizaje máquina para la implementación de una solución
Implementación de un algoritmo de Machine Learning utilizando un framework o librería para la clasificación de un conjunto de datos en python, con el objetivo de realizar predicciones. En estadística, la regresión logística es utilizada para modelar la probabilidad de la existencia de cierta clase o evento. La regresión logísitica mide la relación entre las variables independientes (x) y la variable categórica (y), utilizando una función sigmoide para el cálculo de probabilidades. El modelo es entrenado al dividir el data set utilizado en un subset de entrenamiento y uno de prueba a través de `sci-kit learn`

## Librería 
Librería utilizada para la implementación del modelo de Regresión Logística:
```
sci-kit learn
sklearn.model_selection - LogisticRegression
```


## Dataset utilizado
La base de datos Iris es un dataset multivariable que consiste en 50 muestras de tres especies de Iris, así como algunas propiedades de cada flor. Una de las especies de flor es linealmente separable de las otras dos, pero las otras dos no son linealmente separables entre sí. Las columnas encontradas en el dataset son:
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species
En donde la columna Species es la variable que se busca predecir.
El archivo que contiene el dataset puede ser encontrado en https://www.kaggle.com/uciml/iris. Dentro del repositorio se encuentra tanto el dataset utilizado `iris.data` y un archivo con la descripción del dataset `iris.names`.

## Métrica de desempeño
Las métricas utilizadas para evaluar el desempeño del modelo , utilizando la librería `sklearn. metrics` han sido:
* Accuracy Score
* Coeficiente R2
* Mean Squared Error
