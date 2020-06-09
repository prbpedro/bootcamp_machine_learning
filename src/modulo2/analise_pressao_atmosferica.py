import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

#lendo o dataset para o formato de dataframe
caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
df_poluicao_beijing=pd.read_csv(caminho_base + "PRSA_data_2010.1.1-2014.12.31.csv")

#conhecendo o dataset
print(df_poluicao_beijing.head())

#verificando o shape do dataset
print(df_poluicao_beijing.shape)

#verificando o formato do dataset
df_poluicao_beijing.info()

#contando valores nulos 
print(df_poluicao_beijing.isnull().sum())

#obtendo os dados em formato de datetime
df_poluicao_beijing['datetime']=df_poluicao_beijing[['year', 'month', 'day', 'hour']].apply(lambda row: datetime.datetime(year=row['year'],month=row['month'], day=row['day'],hour=row['hour']), axis=1)

#mostrando a modificação
print(df_poluicao_beijing.head())

#encontrando as datas máximas e mínimas do dataset
print('Data inicial de coleta',df_poluicao_beijing['datetime'].min())
print('Data final de coleta',df_poluicao_beijing['datetime'].max())

df_pressao_atmosferica=df_poluicao_beijing[['datetime','PRES']] #pressão em hPa
print(df_pressao_atmosferica.head())

df_pressao_atmosferica.sort_values('datetime', ascending=True, inplace=True) #ordenando os valores pela data

print(df_pressao_atmosferica.head())

#verificando possíveis outlier com o boxplot
plt.figure(figsize=(7, 7))
g = sns.boxplot(df_pressao_atmosferica['PRES'])
g.set_title('Box plot para a Pressão Atmosférica')
plt.show()

#plotando os valores de pressão atmosférica
plt.figure(figsize=(7, 7))
g = sns.lineplot(x=df_pressao_atmosferica.index,y=df_pressao_atmosferica['PRES'])
g.set_title('Serie temporal para a pressão atmosférica')
g.set_xlabel('Indice do dataset')
g.set_ylabel('Pressão Atmosférica em hPa')
plt.show()

"""**Preparando os dados para serem utilizados no modelo de previsão via MLP**"""

from sklearn.preprocessing import MinMaxScaler  #aplicando a normalização dos dados
scaler = MinMaxScaler(feature_range=(0, 1))  #define o intervalor entre 0 e 1 para os dados serem normalizados
df_pressao_atmosferica['PRES_normalizado'] = scaler.fit_transform(np.array(df_pressao_atmosferica['PRES']).reshape(-1, 1))

print(df_pressao_atmosferica.head())

#dividindo os dados entre treinamento e teste
data_de_corte = datetime.datetime(year=2014, month=1, day=1, hour=0)  #difine a data de corte para 01/01/2014
df_treinamento = df_pressao_atmosferica.loc[df_pressao_atmosferica['datetime']<data_de_corte]
df_teste = df_pressao_atmosferica.loc[df_pressao_atmosferica['datetime']>=data_de_corte]
print('Quantidade de dados para treinamento:', df_treinamento.shape)
print('Quantidade de dados para teste:', df_teste.shape)

#verificando "a cara" dos dados de treinamento 
plt.figure(figsize=(7, 7))
g = sns.lineplot(x=df_treinamento.index,y=df_treinamento['PRES_normalizado'], color='b')
g.set_title('Série Temporal para a pressão normalizada no treinamento do modelo')
g.set_xlabel('Indices')
g.set_ylabel('Leituras normalizadas')
plt.show()

#verificando "a cara" dos dados de teste
plt.figure(figsize=(7, 7))
g = sns.lineplot(x=df_teste.index,y=df_teste['PRES_normalizado'], color='r')
g.set_title('Série Temporal para a pressão normalizada no teste do modelo')
g.set_xlabel('Indices')
g.set_ylabel('Leituras normalizadas')
plt.show()

#definindo o número de valores a serem utilizados para a previsão 
def formata_entrada_saida(serie_temporal_original, numero_de_passos):
    X = []
    y = []
    for i in range(numero_de_passos, serie_temporal_original.shape[0]):
        X.append(list(serie_temporal_original.loc[i-numero_de_passos:i-1]))
        y.append(serie_temporal_original.loc[i])
    X, y = np.array(X), np.array(y)
    return X, y

#formata os dados para o treinamento do modelo
X_train, y_train = formata_entrada_saida(df_treinamento['PRES_normalizado'], 10)
print('Formato dos dados:', X_train.shape, y_train.shape)

print(X_train[1:15,:])

X_teste, y_teste = formata_entrada_saida(df_teste['PRES_normalizado'].reset_index(drop=True),10)
print('Formato dos dados de normalizacao:', X_teste.shape, y_teste.shape)

print(X_teste[1:15,:])

"""**Inicia o processo de construção da Previsão via MLP**"""

#importando as bibliotecas 
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout  #define os tipos de camadas a serem utilizadas pelo modelo
# Camada Dropout - Evita o overfitting retirando aleatoriamente neuronios durante o treinamento
from tensorflow.keras.optimizers import SGD  #define o modelo de otimização via gradiente descendente

#define a camada de entrada
camada_entrada = Input(shape=(10,), dtype='float32')

#adiciona as camadas escondidas
densa1 = Dense(32, activation='linear')(camada_entrada)
densa2 = Dense(16, activation='linear')(densa1)
densa3 = Dense(16, activation='linear')(densa2)

#adiciona a camada de dropout como forma de regularização do modelo (ajuda a evitar overfitting)
camada_dropout = Dropout(0.2)(densa3)

#camada de saída da rede (1 dimensão, pois queremos prever a pressão atmosférica baseada em valores anteriores)
camada_de_saida = Dense(1, activation='linear')(camada_dropout)

#definindo o modelo MLP
modelo_MLP = tf.keras.Model(inputs=camada_entrada, outputs=camada_de_saida)

#mostrando as características do modelo 
modelo_MLP.summary()

#definindo a função de erro e o otimizador a ser utilizado
modelo_MLP.compile(loss='mean_squared_error', optimizer='adam')  #função perda MSE e otimizador de Adam

#treina o modelo
modelo_MLP.fit(x=X_train, y=y_train, batch_size=16, epochs=20,verbose=1, shuffle=True)

#realiza a previsão com o modelo MLP
previsao = modelo_MLP.predict(X_teste)
previsao_PRES = scaler.inverse_transform(previsao)  #aplica o inverso da transformação
print(previsao_PRES.shape)

previsao_PRES = np.squeeze(previsao_PRES)  #remove entradas de uma dimensão
print(previsao_PRES.shape)

from sklearn.metrics import r2_score  #importa o coeficiente de determinação

r2 = r2_score(df_teste['PRES'].iloc[10:], previsao_PRES)
print('Coeficiente de Determinação Para o Teste (MLP):', round(r2,4))

#plotando os valores reais x previstos
plt.figure(figsize=(7,7))
plt.plot(range(50), df_teste['PRES'].iloc[10:60], linestyle='-', marker='*', color='r')
plt.plot(range(50), previsao_PRES[:50], linestyle='-', marker='.', color='b')
plt.legend(['Real','Previsto'], loc=2)
plt.title('Valor de Pressão Atmosferica Medido vs Valor de Pressão Atmosférica Previsto (MLP)')
plt.ylabel('Pressão Atmosférica')
plt.xlabel('Indice')
plt.show()

"""**Inciando o processo de previsão via CNN**"""

from tensorflow.keras.layers import Flatten  #camada flatten para transformar os dados em uma dimensão 
from tensorflow.keras.layers import ZeroPadding1D  #completa os dados após a convolução
from tensorflow.keras.layers import Conv1D  #camada de convolução
from tensorflow.keras.layers import AveragePooling1D  #camada de redução (média dos dados encontrados)

#define a camada de entrada 
camada_entrada = Input(shape=(10,1), dtype='float32')

#adiciona a camada de padding
camada_padding = ZeroPadding1D(padding=1)(camada_entrada)  #matém a quantidade de dados

#adiona a camada de convolução 
camada_convolucao_1D = Conv1D(64, 3, strides=1, use_bias=True)(camada_padding) #adiciona 64 filtros com uma janela de convolução=3

#camada de pooling
camada_pooling = AveragePooling1D(pool_size=3, strides=1)(camada_convolucao_1D)  #reduz através do valor médio encontrado para a convolução (pode ser também o valor máximo)

#camada flatten
camada_flatten = Flatten()(camada_pooling) #utilizada para realizar o "reshape" dos dados para um vetor

#adicionando a camada de dropout
camada_dropout_cnn = Dropout(0.2)(camada_flatten)

#camada de saída
camada_saida = Dense(1, activation='linear')(camada_dropout_cnn)

#contruindo o modelo
modelo_CNN = tf.keras.Model(inputs=camada_entrada, outputs=camada_saida)

#mostrando o modelo
modelo_CNN.summary()

#adionando a função perda e o otimizados
modelo_CNN.compile(loss='mean_absolute_error', optimizer='adam')

#Transforma os dados de treinamento e teste para o 3D, pois a rede CNN exige essa transformação
X_train, X_teste = X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),X_teste.reshape((X_teste.shape[0], X_teste.shape[1], 1))
print('Formatos para o treinamento e teste:', X_train.shape, X_teste.shape)

#realizando o treinamento do modelo
modelo_CNN.fit(x=X_train, y=y_train, batch_size=16, epochs=20,verbose=1,shuffle=True)

#previsão CNN
previsao_cnn = modelo_CNN.predict(X_teste)
PRES_cnn = np.squeeze(scaler.inverse_transform(previsao_cnn))

r2_cnn = r2_score(df_teste['PRES'].iloc[10:], PRES_cnn)
print('Coeficiente de Determinação Para o Teste (CNN):', round(r2, 4))

#plotando os valores reais x previstos
plt.figure(figsize=(7,7))
plt.plot(range(50), df_teste['PRES'].iloc[10:60], linestyle='-', marker='*', color='r')
plt.plot(range(50), PRES_cnn[:50], linestyle='-', marker='.', color='b')
plt.legend(['Real','Previsto'], loc=2)
plt.title('Valor de Pressão Atmosferica Medido vs Valor de Pressão Atmosférica Previsto (CNN)')
plt.ylabel('Pressão Atmosférica')
plt.xlabel('Indice')
plt.show()