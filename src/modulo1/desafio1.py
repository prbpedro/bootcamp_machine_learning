import pandas
from sklearn.preprocessing import LabelEncoder
import seaborn
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn.preprocessing
import sklearn.tree
import sklearn.model_selection

base_path = '/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/'
dataframe = pandas.read_csv(base_path + "src/input/desafio1/comp_bikes_mod.csv")

dataframe.info()
# Resposta 1 - (17379,17)
# Resposta 2 - 2

null_percentage = dataframe.isna().mean().round(4)*100
print(null_percentage)
# Resposta 3 - 10%

df_dteday = dataframe.iloc[:,1] # dteday
dataframe_nonulls_dteday = dataframe.dropna(subset=['dteday'])
dataframe_nonulls_dteday.info()
# Resposta 4 - (15641,17)

print(dataframe_nonulls_dteday.describe())

print(dataframe_nonulls_dteday[['temp']].describe())
# Resposta 5 - 0.496926

print(dataframe_nonulls_dteday[['windspeed']].describe())
# Resposta 6 - 0.122309

a = dataframe_nonulls_dteday["season"].to_numpy()

dataframe["season"] = dataframe["season"].astype("category")
le = LabelEncoder()
le.fit(dataframe["season"].dropna().values)
classes = le.classes_
print(le.classes_)
print(dataframe["season"].cat.categories)
dataframe.info()
# Resposta 7 - 4

df_dt = pandas.to_datetime(dataframe_nonulls_dteday['dteday'])
print(df_dt.max())
# Resposta 8 - 2012-12-31

dataframe_nonulls_dteday.boxplot(column = ["windspeed"])
matplotlib.pyplot.show()
# Resposta 9 - Existem possíveis outliers, pois existem marcações (pontos) foras dos limites do boxplot.

df_reduced = dataframe_nonulls_dteday[['season','temp','atemp','hum','windspeed','cnt']]
matplotlib.pyplot.figure(figsize=(20, 10))
corr_matrix = df_reduced.corr()
seaborn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,center= 0)  
matplotlib.pyplot.show()
# Resposta 10 - Possuem baixa correlação linear negativa.

df_reduced = dataframe_nonulls_dteday.loc[:,['hum','casual','cnt']]
df_reduced.fillna(df_reduced.mean(), inplace=True)
x=df_reduced.loc[:,['hum','casual']]
y=df_reduced.loc[:,'cnt'] 
linear_regression=LinearRegression()
linear_regression.fit(x,y) 
prediction=linear_regression.predict(x)
r2 = r2_score(y, prediction)  
print("Coeficiente de Determinação (R2):", r2, end='\n\n')
# Resposta 11 - 0.40

entrada_arvore=df_reduced[['hum','casual']]
saida_arvore=y=df_reduced['cnt']
regression_tree=sklearn.tree.DecisionTreeRegressor() 
regression_tree.fit(entrada_arvore, saida_arvore)
tree_prediction = regression_tree.predict(entrada_arvore)
r2 = regression_tree.score(entrada_arvore, saida_arvore)
r2_ = r2_score(saida_arvore, tree_prediction)  
print("Coeficiente de Determinação (R2) TREE:", r2, r2_, end='\n\n')
# Resposta 12 - 0.70
# Resposta 13 - O valor obtido pela árvore de decisão como regressor apresenta maior R2
# Resposta 14 - Pode ser utilizada para classificação e regressão.
# Resposta 15 - SVM encontra o hiperplano que gera a maior separação entre os dados.
