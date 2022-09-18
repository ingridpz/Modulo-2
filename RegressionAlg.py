import pandas as pd
import numpy as np
import math

#Cargar los datos 
df = pd.read_csv("C:/Users/Antos/Documents/R/iris (1).data")
df.columns = ["sepal length", "sepal width", "petal length", "petal width","species"]


#Label Encoding
df.replace('Iris-versicolor', "Iris-versicolour", inplace = True)
df.replace('Iris-setosa',0, inplace =True)
df.replace('Iris-versicolour', 1, inplace = True)
df.replace('Iris-virginica',2, inplace =True)
pd.to_numeric(df["species"])
#El algoritmo será de clasificación binaria por lo que se remueve una clase para predecir
df = df[df['species'] <2]

#Función para dividir el dataset en entrenamiento y prueba
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[test_indices], data.iloc[train_indices]

#Dividir el dataset
test, train = split_train_test(df, 0.2)
x_test = test.drop(['species'], axis=1)
y_test = test['species']
x_train = train.drop(['species'], axis=1)
y_train = train['species']

#Algoritmo de Regresión Logística
def RegresionLogistica(X, y, a, n_iters):
    theta = np.zeros(X.shape[1])
    n = y.size
    J = np.zeros(n_iters)
    for i in range(n_iters):
        # Calcular la hipótesis
        h = sigmoid(X.dot(theta))
        # Calcular el gradiente
        gradiente = (1/n)*X.T.dot(h-y)
        # Actualizar los parámetros
        theta -= a * gradiente
        # Calcular la función de costo
        J[i] = FuncionCosto(theta, X, y)
    return theta, J

#Función sigmoide
def sigmoide(x):
    h = 1/(1+np.exp(-x))
    return h

#Función de costo
def FuncionCosto(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = (1/m)*(-y.dot(np.log(h))-(1-y).dot(np.log(1-h)))
    return J

#Entrenar el modelo
a=0.001
n_iters = 1000
theta, J =RegresionLogistica(x_train, y_train,a, n_iters )
print(theta)

#Generar predicciones
def predict(theta, X):
    predictions = sigmoid(X.dot(theta))
    predictions= [1 if i > 0.5 else 0 for i in predictions]
    return predictions

predictions = predict(theta, x_test)
preds = predictions
for a in range(0, len(preds)):
    if preds[a] == 0:
        preds[a] = "Iris-Setosa"
    elif preds[a]==1:
        preds[a] = "Iris-Versicolour"
print("Predicciones \n", preds)
y_test2 = y_test.copy()
y_test2.replace(0,'Iris-setosa', inplace =True)
y_test2.replace(1,'Iris-versicolour', inplace=True)
print("Valor esperado: \n", y_test2)


#Métricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
accuracy = accuracy_score(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("Accuracy Score: ", accuracy)
print("R2 score: ", r2)