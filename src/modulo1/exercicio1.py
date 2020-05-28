#importando as bibliotecas
import pandas as pd #biblioteca utilizada para o tratamento de dados via dataframes 
import numpy as np #biblioteca utilizada para o tratamento de valores numéricos (vetores e matrizes)
import matplotlib.pyplot as plt #biblioteca utilizada para construir os gráficos

dataframe_envio_portos= pd.read_csv("src/input/exercicio1/data.csv")

#apresentando as 5 primeiras linhas do dataset
dataframe_envio_portos.head()

dataframe_envio_portos.info() #verificando os tipos de variáveis e se existem ou não valores nulos

"""**Existem Colunas Com Valores Nulos?**"""

dataframe_envio_portos.shape

"""**Quantas Instâncias e Características Existem no Dataset?**"""

#analisando a "estatística" do dataset
dataframe_envio_portos.describe()

"""**Qual é o Valor Médio Para os Pesos Declarados?**"""

#identificando possíveis outliers
dataframe_envio_portos[['declared_quantity','days_in_transit']].boxplot()

"""**Existem Possíveis Outliers?**"""

#realizando a análise de regressão
x=dataframe_envio_portos['declared_weight'].values  #variável independente 
Y=dataframe_envio_portos['actual_weight'].values #variável dependente

#importa o modelo de regressão linear univariada
from sklearn.linear_model import LinearRegression

#Realiza a construção do modelo de regressão
reg= LinearRegression()
x_Reshaped=x.reshape((-1, 1)) #coloca os dados no formato 2D
regressao= reg.fit (x_Reshaped,Y) # encontra os coeficientes (realiza a regressão)
print(regressao)

#realiza a previsão
previsao=reg.predict(x_Reshaped)

#análise do modelo
from sklearn.metrics import r2_score #método para o cálculo do R2 (coeficiente de determinação)

#parâmetros encontrados
print('Y = {}X {}'.format(reg.coef_,reg.intercept_))

R_2 = r2_score(Y, previsao)  #realiza o cálculo do R2

print("Coeficiente de Determinação (R2):", R_2)

"""**Pelo Coefiente de Determinação, o Que É Possível Afirmar Sobre a Relação Entre as Variáveis Peso Real x Peso Declarado?**"""

#realiza o plot dos dados
plt.figure(figsize=(4, 4), dpi=100)
plt.scatter(x, Y,  color='gray') #realiza o plot do gráfico de dispersão
plt.plot(x, previsao, color='red', linewidth=2) # realiza o plto da "linha"
plt.xlabel("Peso Declarado")
plt.ylabel("Peso Real")
plt.show()