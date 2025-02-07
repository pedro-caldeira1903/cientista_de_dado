import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
dados=pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv")
dados=dados.drop(columns='Unnamed: 0', axis=0)
dados=dados.drop(columns='idade_do_modelo', axis=0)
x=dados.drop(columns='vendido', axis=0)
y=dados['vendido']
treino_x, teste_x, treino_y, teste_y=train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
modelo=Pipeline([('scal', StandardScaler()), ('model', DecisionTreeClassifier(max_depth=4))])
modelo.fit(treino_x, treino_y)
predicao=modelo.predict(teste_x)
print(accuracy_score(teste_y, predicao)*100)
