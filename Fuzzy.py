"""
Created on Tue Oct 20 23:11:43 2020

@author: LuisInnocencio
"""

import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import classification_report

#Dimensionamento das variaveis 
Lpetala = ctrl.Antecedent(np.arange(0, 3.5, 0.1), 'Lpetala')
Cpetala = ctrl.Antecedent(np.arange(1, 8, 0.5), 'Cpetala')
tipo = ctrl.Consequent(np.arange(0, 3, 1), 'tipo')


#Definição das atribuições de cada variavel
Lpetala.automf(names=['pp','p','m','m2', 'g' ,'mg','gg'])

Cpetala.automf(names=['pp', 'p','m','g','mg'])

tipo['0'] = fuzz.trimf(tipo.universe, [0, 0, 1])
tipo['1'] = fuzz.trimf(tipo.universe, [0.5, 1, 1.5])
tipo['2'] = fuzz.trimf(tipo.universe, [1.5, 2, 2])


#Regras da defuzzificação 
rule1=ctrl.Rule(Lpetala['p'], tipo['0'])
rule2=ctrl.Rule(Lpetala['m'] & (Cpetala['pp']|Cpetala['p']|Cpetala['m']) , tipo['1'])
rule3=ctrl.Rule((Lpetala['g']|Lpetala['mg']|Lpetala['gg']|Lpetala['m2']) & (Cpetala['g']|Cpetala['mg']) , tipo['2'])




#Comandos de simulação
tipo_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
tipo_simulador = ctrl.ControlSystemSimulation(tipo_ctrl);


#Abertura do datasheet
iris = load_iris() 
df_iris = pd.DataFrame(np.column_stack((iris.data, iris.target)), 
    columns = iris.feature_names + ['target'])
df_iris.describe()
#df_iris.plot()
X=[]


for o in range(0,150,1):
    print(o)
    
    a=df_iris.loc[[o],['petal width (cm)']]
    b=df_iris.loc[[o],['petal length (cm)']]

    c= a.values
    d= b.values

#Entrada de dados
    tipo_simulador.input['Lpetala'] = float(np.asarray(c))
    tipo_simulador.input['Cpetala'] = float(np.asarray(d))

#Inicio de simulação

    tipo_simulador.compute()
    if (tipo_simulador.output['tipo']<0.75):
        X.append(0)
    if (tipo_simulador.output['tipo']<1.55):
            if tipo_simulador.output['tipo']>0.75:
                X.append(1)
                
    if tipo_simulador.output['tipo']>1.55:
                    X.append(2)
    
    
u=iris.target
t=u.reshape(150,1)
r = np.array(X)

print(classification_report(r, t))
    
#Visualização grafica
Lpetala.view(sim=tipo_simulador)
Cpetala.view(sim=tipo_simulador)
tipo.view(sim=tipo_simulador)





#Carrega o iris dataset em iris 
#
#Cria o DataFrame em df_iris utilizando um numpy array (np) 


Lpetala.view()
Cpetala.view()
tipo.view()