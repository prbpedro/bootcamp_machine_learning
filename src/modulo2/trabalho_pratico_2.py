#importando as bibliotecas
import pandas as pd  #bibioteca responsável para o tratamento e limpeza dos dados
from matplotlib import pyplot as plt  #plotar os gráficos
import seaborn as sns #plot de gráficos

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
#carregando os dados para o pandas
df_consultas=pd.read_csv(caminho_base + 'KaggleV2-May-2016.csv')

"""**Iniciando a análise exploratória**"""

#mostrando as características do dataset
print(df_consultas.head(10))

#mostrando as dimensões do dataset
print(df_consultas.shape)

"""**Quantas instâncias e atributos existem no dataset?**"""

#mostrando as "características" das colunas
df_consultas.info()

#contando a quantidade de valores nulos
print(df_consultas.isnull().sum())

"""**Existem valores nulos?**"""

print(df_consultas['No-show'].value_counts()) # Yes o paciente não compareceu / No o paciente compareceu

print(df_consultas['No-show'].value_counts()['No']/len(df_consultas['No-show']))

#analisando as "estatísticas do dataset"
df_consultas.describe()

"""**Qual é a idade média dos pacientes?**"""

#contando a quantidade de valores distintos em cada uma das colunas
for colunas in list(df_consultas.columns):
  print( "{0:25} {1}".format(colunas, df_consultas[colunas].nunique()) )

"""**Em quantas localidades diferentes (Neighbourhood) os pacientes residem?**

**Comparando a identificação do paciente (PatientId) com o número dos agendamentos das consultas (AppointmentID) o que podemos inferir?**

**Quantas variáveis binárias (apenas dois valores) existem no dataset?**
"""

#contando quantas idades diferentes existem no df
print(df_consultas['Age'].nunique())

#plotando o histograma de algumas variáveis 
df_consultas['Age'].hist(bins=len(df_consultas['Age'].unique()))

df_consultas['SMS_received'].hist(bins=len(df_consultas['SMS_received'].unique()))

print('SMS_received unique: ', df_consultas['SMS_received'].unique())


print(df_consultas['SMS_received'].value_counts())

"""**Quantos valores diferentes encontramos para a vairável dos SMS recebidos (SMS_received)?**"""

#criando uma nova coluna -> Tempo de espera (diferença entre a data em que a consulta foi agendada e o dia da consulta)
df_consultas.ScheduledDay=pd.to_datetime(df_consultas.ScheduledDay)  #transformando as colunas par o tipo datetime
df_consultas.AppointmentDay=pd.to_datetime(df_consultas.AppointmentDay)

#Encontra a diferença entre o momento da marcação da consulta e o dia da consulta
tempo_espera=df_consultas.ScheduledDay-df_consultas.AppointmentDay

print(tempo_espera[:10])

df_consultas['AwaitingTime']=tempo_espera.apply(lambda x: x.days) #transforma os valores em dias

print(df_consultas.head(8))

df_consultas.info()

"""**Iniciando o tratamento dos dados**

**Qual é a menor e maior idade, respectivamente, presente no dataset?**
"""
print(df_consultas['Age'].mean())

#Encontrando as idades negativas
print(df_consultas[df_consultas['Age'] < 0]['Age'].value_counts())

#filtrando apenas idades maiores que 0
df_consultas_filtrado=df_consultas[df_consultas['Age']>=0]
print(df_consultas_filtrado.shape)

"""**Quantos valores de idade menores do que 0 existem no dataframe?**"""

#transformando os tempo de espera para um valor não negativo
df_consultas_filtrado['AwaitingTime'] = df_consultas_filtrado['AwaitingTime'].apply(lambda x: abs(x))

print(df_consultas_filtrado.head())

#aplicando a transformação para os dados categóricos
categoricas=['Neighbourhood','Gender','No-show']
for coluna in categoricas:
  df_consultas_filtrado[coluna]=pd.Categorical(df_consultas_filtrado[coluna]).codes

print(df_consultas_filtrado.head(8))

"""**Qual o tipo de transformação foi utilizada?**"""

#analisando os SMS enviados e o número de vezes que o paciente compareceu ou não 
sms_x_comparecer = df_consultas_filtrado.groupby(['SMS_received', 'No-show'])['SMS_received'].count().unstack('No-show').fillna(0)
sms_x_comparecer[[0, 1]].plot(kind='bar', stacked=True) 
plt.title('Analise do número de SMS recebido e se a paciente compareceu ou não à consulta') 
plt.xlabel('Numero de SMS recebidos') 
plt.ylabel('Frequência')
plt.show()

"""**Qual é a proporção de pacientes que receberam o sms e NÃO compareceram?**"""

#plotando o número de consultas por região 
regioes = df_consultas_filtrado['Neighbourhood'].unique()
plt.figure(figsize=(22,10))
ax = sns.countplot(x='Neighbourhood', data=df_consultas_filtrado, order=regioes)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
plt.title('Consultas por região', fontsize=14, fontweight='bold')
plt.show()

"""**Qual região possui o maior número de consultas marcadas?**"""

#selecionando os dados para a construção da previsão
entradas = ['Gender','Age','Neighbourhood','Scholarship','Hipertension','Diabetes','Alcoholism','SMS_received','AwaitingTime']
saida=['No-show']

x=df_consultas_filtrado[entradas]
y=df_consultas_filtrado[saida]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

print(x_train.shape)

#aplicando um modelo de classificação via árvore de decisão
from sklearn.tree import DecisionTreeClassifier
clf_arvore = DecisionTreeClassifier() 
clf_arvore.fit(x_train, y_train)

#realiza a previsão com os dados
y_previsto = clf_arvore.predict(x_test)

from sklearn.metrics import accuracy_score
acuracia = accuracy_score(y_test, y_previsto)
print('Acurácia da àrvore de Decisão: ',acuracia)

#contrução da matriz de confusão
from sklearn.metrics import confusion_matrix
matriz_confusao = confusion_matrix(y_test, y_previsto)

#realiza o plot da matriz de confusão
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao)
plt.show()