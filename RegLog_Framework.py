import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Importar el dataset
df = pd.read_csv("C:/Users/Antos/Documents/R/iris (1).data")
df.columns = ["sepal length", "sepal width", "petal length", "petal width","species"]

#Calcular correlación
corr = df.corr()

#Dividir el dataset
x = df.drop(["species"], axis=1)
y = df['species']
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.15, random_state=42)

#transformación de la variable species en valores numéricos
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])
ytest = le.transform(ytest)

#Regresión logística
from sklearn.linear_model import LinearRegression
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

model = LinearRegression()
lrm =model.fit(xtrain, ytrain)
ypred = lrm.predict(xtest)
print("Coeficients and Interception: ",lrm.coef_, lrm.intercept_)

#Métricas
accuracy = accuracy_score(ytest, ypred.round())
r2 = r2_score(ytest, ypred)
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
print("Accuracy: ", accuracy)
print("R2 score: ", r2)
print ("Mse: ", mse)
print("Rmse: ", rmse)

#Validación 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, xtest, ytest, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())

# Predicciones
preds = ypred[0:9]
print(preds.round())
print(ytest[0:9])