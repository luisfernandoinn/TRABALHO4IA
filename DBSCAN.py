# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: João Victor
"""
from sklearn import metrics
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report
import numpy as np

iris = load_iris()
modelo = iris.target
dbscan = DBSCAN()
dbscan.fit(iris.data)
resultado = dbscan.labels_

resultado2=np.zeros(150)+2

for i in range (150):
    if resultado[i]!=(-1):
        resultado2[i]=resultado[i]
       
       
print(classification_report(modelo, resultado2))