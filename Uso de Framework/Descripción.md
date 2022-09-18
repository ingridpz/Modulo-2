# Portafolio de Implementación: Uso de framework de aprendizaje máquina para la implementación de una solución
Implementación de un algoritmo de Machine Learning utilizando un framework o librería para la clasificación de un conjunto de datos en python, con el objetivo de realizar predicciones. En estadística, la regresión logística es utilizada para modelar la probabilidad de la existencia de cierta clase o evento. La regresión logísitica mide la relación entre las variables independientes (x) y la variable categórica (y), utilizando una función sigmoide para el cálculo de probabilidades. El modelo es entrenado al dividir el data set utilizado en un subset de entrenamiento y uno de prueba a través de `sci-kit learn`, el modelo se encuentra en el archivo `RegLog_Framework.py`.

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

Se utilizó 85% de los datos para el entrenamiento del modelo y un 15% para probarlo. Los resultados obtenidos fueron los siguientes:

| Métrica | Valor |
| --------------- | ------ |
| Accuracy Score  | 0.8695 |
| R2| 0.8907 |
| Mse|  0.0656  |

## Resultados
A partir de la implementación del modelo de Regresión Logística, se obtuvieron las siguientes predicciones:
| id | Valor Esperado | Predicción |
| ---|----------------| ------ |
| 1  |Iris-Versicolour| Iris-Versicolour   |
| 2| Iris-Setosa |Iris-Setosa|
| 3|  Iris Virginica |Iris Virginica |
| 4 | Iris-Versicolour |  Iris-Versicolour  |
| 5  |Iris-Versicolour | Iris Virginica  |
| 6  | Iris-Setosa |   Iris-Setosa |
| 7  | Iris-Versicolour |   Iris-Versicolour|
| 8  | Iris Virginica |  Iris Virginica |
| 9  |Iris-Versicolour |   Iris-Versicolour |
| 10  | Iris-Versicolour |   Iris Virginica |
