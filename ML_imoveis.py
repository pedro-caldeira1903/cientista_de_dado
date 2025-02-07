import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
dados=pd.read_json('/content/imoveis.json')
dados1=pd.json_normalize(dados['ident'])
dados2=pd.json_normalize(dados['listing'], sep=' ')
dados3=pd.concat([dados1, dados2], axis=1)
dados3=dados3.drop(columns=['address zone','types unit','customerID','source','types usage','address city','address location lon','address location lat'], axis=0)
dados3.isnull().sum()
dados3=dados3.astype({'prices price': 'float64','prices tax iptu': 'float64','prices tax condo': 'float64','features usableAreas': 'int64','features totalAreas': 'int64'})
dados3=dados3.rename({'address neighborhood': 'bairro','prices price': 'preço','prices tax iptu': 'iptu','prices tax condo': 'condominio','features bedrooms': 'quartos','features bathrooms': 'banheiros','features suites': 'suites','estacionamento': 'estacionamento','area_usada': 'area_usada','metros_quadrados': 'metros_quadrados','piso': 'piso','unidade_piso': 'unidade_piso','andar_piso': 'andar_piso'}, axis=1)
dados3=pd.get_dummies(dados3, columns=['bairro'])
dados3['iptu'].fillna(0, inplace=True)
dados3['condominio'].fillna(0, inplace=True)
dados3=dados3.astype({'bairro_Vigário Geral': 'int64','bairro_Vila Isabel': 'int64','bairro_Vila Kosmos': 'int64','bairro_Vila Militar': 'int64','bairro_Vila Valqueire': 'int64','bairro_Vila da Penha': 'int64','bairro_Vista Alegre': 'int64','bairro_Zumbi': 'int64','bairro_Água Santa': 'int64','bairro_Área Rural de Rio de Janeiro': 'int64'})
x=dados3.drop(columns='preço', axis=1)
y=dados3['preço']
treino_x, teste_x, treino_y, teste_y=train_test_split(x, y, test_size=0.3, random_state=42)
modelo=Pipeline([('scal', StandardScaler()), ('model', DecisionTreeClassifier())])
modelo.fit(treino_x, treino_y)
predicao=modelo.predict(teste_x)
print(f'{r2_score(predicao, teste_y)*100}')
