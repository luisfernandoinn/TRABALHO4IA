from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

iris_data = load_iris() # Carrega o dataset
# Após isso criamos uma tabela usando a biblioteca Pandas para dividir o dataset em dois um para treino e outro como target
X, y = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names), pd.DataFrame(data=iris_data.target, columns=["iris_quality_type"])
# Aqui chamamos uma função para testar e dividir o dataset em valores aleatórios com a finalidade de analisar o comportamento do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.30)
#Aqui chamamos uma função da biblioteca NumPy para criar uma copia da matriz de dados , no caso. dividindo o data set
y_train, y_test = np.ravel(y_train), np.ravel(y_test)
# Implementamos a função da biblioteca sklearn com a finalidade de encaixar os dados no modelo KNN, no caso com 3 vizinhos
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Função que retorna valores através de uma predição de dados aleatórios
y_pred = knn.predict(X_test)
#print(y_pred)
# Função que mostra os valores da acurácia do teste
#print(knn.score(X_test, y_test))
# Aqui vamos criar uma função para testar a quantidade de vizinhos e qual a melhor quantidade, no caso criamos um vetor de 1 a 20
neighbors = np.arange(1, 20)
#Criamos duas listas para os valores 
train_accuracy, test_accuracy = list(), list()


for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
    
    
print(classification_report(y_test, y_pred))


plt.figure(figsize=[13, 8])
plt.plot(neighbors, test_accuracy, label="Desempenho do teste")
plt.plot(neighbors, train_accuracy, label="Desempenho")
plt.legend()
plt.title("Valor vs Precisão")
plt.xlabel("Número de vizinhos")
plt.ylabel("Precisão")
plt.xticks(neighbors)
plt.show()

#print("A melhor precisão é {} quando temos o K={}".format(np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))))