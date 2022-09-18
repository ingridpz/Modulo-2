# Portafolio de Implementación: Uso de framework de aprendizaje máquina para la implementación de una solución
Implementación de un algoritmo de Machine Learning sin utilizar un framework o librería para la clasificación de un conjunto de datos en python, con el objetivo de realizar predicciones.
Se implementó un algoritmo de Regresión Logística, el cual es una técnica de aprendizaje supervisado, utilizado para la clasificación. Este algoritmo estima la probabilidad de que un evento ocurra. La implemantación
del algoritmo de Regresión Logística se encuentra en el archivo `RegressionAlg.py`
## Dataset utilizado
La base de datos Iris es un dataset multivariable que consiste en 50 muestras de tres especies de Iris, así como algunas propiedades de cada flor. Una de las especies de flor es linealmente separable de las otras dos, pero las otras dos no son linealmente separables entre sí. Las columnas encontradas en el dataset son:
* Sepal Length
* Sepal Width
* Petal Length
* Petal Width
* Species

En donde la columna Species es la variable que se busca predecir. En el dataset se encuentra información de tres clases diferentes:
* Iris Setosa
* Iris Versicolour
* Iris Virginica

El archivo que contiene el dataset puede ser encontrado en https://www.kaggle.com/uciml/iris. Dentro del repositorio se encuentra tanto el dataset utilizado `iris.data` y un archivo con la descripción del dataset `iris.names`.
El objetivo del modelo es clasificar a partir de los atributos encontrados en el dataset, es decir, predecir la especie de flor. Debido a que el modelo es utilizado para clasificación binaria, se eligieron dos de las clases para entrenar el modelo. Para la implementación del Modelo se utilizaron dos de estas especies: Iris Setosa e Iris Versicolour. Parte del procesamiento de los datos previo a la implementación del modelo ha sido
transformar el tipo de datos contenidos en la clase, por lo que se sustituyó dentro del dataframe la especie Iris Setosa con un valor de 0 y la especie Iris Versicolour con un valor de 1.
Se implementó una función llamada `split_train_test` para dividir el conjunto de datos en el conjunto de entrenamiento y el de pruebas.




## Métrica de desempeño
Las métricas utilizadas para evaluar el desempeño del modelo , utilizando la librería `sklearn. metrics` han sido:
* Accuracy Score
* Coeficiente R2

Se utilizó 80% de los datos para el entrenamiento del modelo y un 20% para probarlo. Los resultados obtenidos fueron los siguientes:

| Métrica | Valor |
| --------------- | ------ |
| Accuracy Score  | 0.95 |
| R2| 0.85294 |


## Resultados
Se obtuvieron 14 predicciones correspondientes al conjunto de pruebas, los resultados obtenidos en comparación con los resultados esperados han sido los siguientes:
| id | Valor Esperado | Predicción |
| ---|----------------| ------ |
| 1  |Iris-Versicolour|Iris-Versicolour  |
| 2| Iris-Setosa |Iris-Setosa|
| 3|  Iris-Setosa |Iris-Setosa |
| 4 |Iris-Versicolour| Iris-Versicolour|
| 5  | Iris-Versicolour |  Iris-Versicolour  |
| 6  |Iris-Versicolour |  Iris-Versicolour |
| 7  |Iris-Versicolour |   Iris-Versicolour|
| 8  | Iris-Setosa |   Iris-Setosa |
| 9  | Iris-Setosa|   Iris-Setosa|
| 10  |Iris-Setosa |  Iris-Setosa |
| 11 | Iris-Setosa|   Iris-Setosa |
| 12  |Iris-Versicolour | Iris-Versicolour|
| 13  | Iris-Versicolour |  Iris-Setosa |
| 14  | Iris-Versicolour|   Iris-Versicolour|
