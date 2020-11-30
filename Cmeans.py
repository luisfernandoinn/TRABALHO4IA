# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
#Carrega o iris dataset de treinamento em iris
iris = load_iris()
# Iremos dividir o data set em dois, o primeiro é o dataset de dados e o segundo de target,
# que são os dados que queremos encontrar 
modelo = iris.target
dados = iris.data
# A seguir, iremos definir o nosso modelo. Em que a biblioteca já foi mensionada acima.
kmeans = KMeans(n_clusters = 3)
 #definindo 3 clusters ou k's
KMmodel = kmeans.fit(dados)
resultado = KMmodel.labels_


resultado2=np.zeros(150)+1

for i in range (150):
    if resultado[i]!=(1):
        resultado2[i]=resultado[i]

print(classification_report(modelo, resultado2))


# Para calcular a porcentagem de acertos iremos utilizar o metódo do classification_report que nos mostra os KPI's de desempenho

"""

# Generate uniformly sampled data spread across the range [0, 10] in x and y
newdata = escolher os dados a serem testados

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    newdata.T, cntr, 2, error=0.005, maxiter=1000)
"""
