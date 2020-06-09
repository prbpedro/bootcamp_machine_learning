import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dados_completos=pd.read_csv(caminho_base + 'airline-passengers.csv')

print(dados_completos.head())
dados_completos.info()

dados_completos['datetime']=pd.to_datetime(dados_completos['Month'])

print(dados_completos.head())
dados_completos.info()

#verificando o tipo de dados
print(type(dados_completos))  

#verificando se existem outliers
plt.figure(figsize=(10, 10))
g = sns.boxplot(dados_completos['Passengers'])  #realiza o plot através da biblioteca seaborn
g.set_title('Box plot para o embarque passageiros')

#plotando o gráfico da variação do número de passageiros no período
plt.figure(figsize=(20, 10))
g = sns.lineplot(x=dados_completos.index,y=dados_completos['Passengers'])
g.set_title('Série Temporal do embarque de passageiros')
g.set_xlabel('Índice')
g.set_ylabel('Número de passageiros em viagens de avião')



#realizando a decomposição da série temporal
#Para encontrar a tendência, sazonalidade e ruído do modelo
#biblioteca responsável por realizar a decomposição da série temporal
from statsmodels.tsa.seasonal import seasonal_decompose  

#modificando o indice para ser temporal
df_serie_temporal=dados_completos.set_index('datetime')

#verifica as colunas existentes
print(df_serie_temporal.columns)

#realiza o drop da coluna 'month'
df_serie_temporal.drop('Month',axis=1,inplace=True)

#verifica o novo dataset
print(df_serie_temporal.head())

#realizando a construção do modelo de decomposição da série temporal
#aplica o modelo de decomposição aditiva
decomposicao_aditiva = seasonal_decompose(df_serie_temporal, model='aditive',extrapolate_trend='freq')   

#realiza o plot da decomposição
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
fig=decomposicao_aditiva.plot()  #realiza o plot da decomposição
plt.show()

#testando a estacionariedade da série temporal
#importando o teste ADF
from statsmodels.tsa.stattools import adfuller  

#aplica o teste adf
resultado_ADF = adfuller(df_serie_temporal.Passengers.values, autolag='AIC')
#para o teste ADF a hipótese nula é que existe, pelo menos, uma raiz negativa 
#na série temporal (série é não-estacionária)

# com o p-valor maior que 0,05 a hipótese nula não é rejeitada
# > 0.5 estacionária
# < 0.5 não estacionária
print('ADF P-valor:',resultado_ADF[1] )


#retirando a tendência
detrended = decomposicao_aditiva.resid + decomposicao_aditiva.seasonal
plt.plot(detrended)
plt.show()  

#retirando a sazonalidade
deseasonalized = decomposicao_aditiva.trend + decomposicao_aditiva.resid
plt.plot(deseasonalized)
plt.show()  

#realizando a análise de autocorrelação nos dados
#importando a biblioteca para o plot da autocorrelação
from statsmodels.graphics.tsaplots import plot_acf   

#aplica a autocorrelação entre os dados
plot_acf(df_serie_temporal, lags=50)  
#mostra uma correlação significativa com 14 lags
plt.show()  

#Transformando a série em estacionária
#aplica o primeiro "Shift" (derivada para tempo discreto)
df_serie_temporal['Passengers_diff'] = df_serie_temporal['Passengers'] - df_serie_temporal['Passengers'].shift(1)  

#retira os valores nulos
df_serie_temporal['Passengers_diff']=df_serie_temporal['Passengers_diff'].dropna()  

df_serie_temporal['Passengers_diff'].plot()
plt.show()  

#Conferindo se agora está estacionária
X_diff = df_serie_temporal['Passengers_diff'].dropna().values
resultado_primeira_diff = adfuller(X_diff)

#pvalor, praticamente 0.05, não rejeita a hipótese nula, mas vamos considerar que está estacionária
print('p-valor: %f' % resultado_primeira_diff[1]) 


#bibliotecas utilizadas para a construção dos modelos de previsão de vendas de passagens
import numpy
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # Camada de conexão enter neuronios
from tensorflow.keras.layers import LSTM # 
from sklearn.preprocessing import MinMaxScaler

#volta o dataset para o formato original
serie_passageiros=df_serie_temporal['Passengers'].values

# normalização do banco de dados, necessário para que os algoritmos possam ter um comportamento mais "previsível"
#cria o objeto que realiza a normalização dos dados por meio dos valores mínimos e máximos
scaler = MinMaxScaler(feature_range=(0, 1))

 # aplica a escala
dataset = scaler.fit_transform(serie_passageiros.reshape(-1, 1))

print(dataset[0:20])

# Divide o conjunto de dados em treinamento e teste 
train_size = int(len(dataset) * 0.67)  #encontra o valor máximo para o treinamento
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test)) #tamanho do df para treinamento e teste

#Cria a matriz necessária para a entrada de dados 
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

#cria o reshape para que os dados estejam em um formato ideal para entrada
look_back = 14  # será utilizado apenas um passo anterior para a previsão do futuro
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape

# cria o modelo utilizando redes recorrentes e o LSTM
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

#apresenta a arquitetura da rede
model.summary()

#realiza o treinamento o modelo de previsão
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# realiza as previsões
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# coloca os dados no formato original
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Realiza a mudança dos dados para a previsão
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#shift para os dados de teste
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Realiza o plot dos dados de previsão e o real
plt.plot(scaler.inverse_transform(dataset),label='Dataset')
plt.plot(trainPredictPlot, label='Treinamento')
plt.plot(testPredictPlot,label='Previsão')
plt.xlabel("Tempo")
plt.ylabel("Número de Passagens Vendidas")
plt.legend()
plt.show()