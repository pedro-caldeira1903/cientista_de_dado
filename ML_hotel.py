import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
dados=pd.read_csv('https://raw.githubusercontent.com/allanspadini/dados-com-muitas-dimensoes/main/dados/hotel.csv')
x=dados.drop(columns='booking_status', axis=1)
y=dados['booking_status']
treino_x, teste_x, treino_y, teste_y=train_test_split(x, y, test_size=0.3, random_state=50, stratify=y)
modelo=Pipeline([('scal', StandardScaler()), ('model', RandomForestClassifier(random_state=2))])
modelo.fit(treino_x, treino_y)
predicao=modelo.predict(teste_x)
print(f'A taxa de acerto do é: {modelo.score(teste_x, teste_y)*100:.2f}')
matriz=confusion_matrix(teste_y, predicao)
disp=ConfusionMatrixDisplay(confusion_matrix=matriz)
disp.plot();
