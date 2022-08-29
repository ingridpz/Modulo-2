#!/usr/bin/env python
# coding: utf-8

# In[62]:


#Regresión Logística

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[63]:


#Cargar los datos
df = pd.read_csv("C:/Users/Antos/Downloads/data.csv")


# In[64]:


df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
y = df['diagnosis']
x = df.drop(['diagnosis'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


# In[65]:


def sigmoide(x):
    h = 1/(1+ np.exp(-x))
    return h


# In[66]:


#Algoritmo de Regresión Logística
def RegresionLogistica (x,y):
    a = 0.001 #learning rate
    iters = 1000 #iteraciones
    n_muestras, n_variables = x.shape
   

    weights = np.zeros(n_variables) #vector de pesos
    t0 = 0 #theta 0

    for k in range(iters):
        xl = np.dot(x, weights)+t0 #vector con los pesos y  bias
        h = 1/(1+ np.exp(-xl))  #aplicar la función sigmoide
        
        #gradiente descendente
        dw = (1/n_muestras)*np.dot(x.T,(h-y)) #derivada de theta j
        dt0 = (1/n_muestras)*np.sum(h-y) #derivada de theta 0
        
        weights -= a*dw  #actualizar los pesos
        t0 -= a*dt0 #actualizar theta 0    


# In[67]:


#función para determinar las precicsión del modelo
def accuracy(y, y_r):
    accuracy = np.sum(y == y_r) / len(y)
    return accuracy


# In[68]:


#Ajustar el modelo
RegresionLogistica(x_train, y_train)


# In[73]:


#Hacer predicciones
n1, n2 = x_test.shape
wpred = np.zeros(n2) #vector de pesos
x_pred= np.dot(x_test, wpred ) + 0
pred = sigmoide(x_pred)
pred= [1 if i > 0.5 else 0 for i in pred]


# In[74]:


accuracy(y_test,pred)


# In[ ]:




