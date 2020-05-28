#importando as bibliotecas
import pandas as pd #biblioteca utilizada para tratar os dados em formato de dataframe
import numpy as np # biblioteca utilizada para tratar vetores e matrizesimport matplotlib.pyplot as plt  #utilizapa para construir os gráficos em um formato similar ao "Matlab"
from sklearn.preprocessing import MinMaxScaler, LabelEncoder #utilizada para realizar o preprocessamento dos dados
from sklearn.model_selection import train_test_split #utilizada para realizar o divisão entre dados para treinamento e teste
from sklearn.metrics import confusion_matrix, accuracy_score #utilizada para verificar a acurácia do modelo construído
from sklearn.naive_bayes import GaussianNB # utilizada para construir o modelo de classificação naive_bayes
import seaborn as sns #utilizada para constuir os gráficos em uma forma mais "bonita"
import matplotlib.pyplot as plt #biblioteca para realizar a construção dos gráficos
from sklearn.svm import SVC #utilizada para importar o algoritmo SVM

base_path = '/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/'
dataset = pd.read_csv(base_path + "src/input/helthcare/data.csv") #realiza a leitura do banco de dados

#print do dataset
print(dataset.head()) # são 76 colunas, mas nem todas serão utilizadas para realizar a previsão de doença cardíaca

print(dataset.shape) # mostra a dimensão do dataset

#conhecendo o dataset
dataset.info()

"""**Existem vários dados nulos**"""

#tratando os dados nulos
dataset.fillna(dataset.mean(), inplace=True) #substitui os dados que estão como NAN pela média dos valores na coluna
print(dataset.head())

"""**Preparando os dados**"""

dataset_to_array = np.array(dataset) #transforma o dataframe em array para facilitar a escolha dos dados a serem utilizados

target = dataset_to_array[:,57] # esse é o vetor de saída (target)
target= target.astype('int') #indica o tipo de dados
#target[target>0] = 1 # 0 para o coração saudável e 1 para problema detectado
print(target)

"""**Iniciando a previsão**"""

#dados coletados pelos sensores
dataset_sensor = np.column_stack(( 
    dataset_to_array[:,11], # pressão sanguínea em repouso
    dataset_to_array[:,33], # frequencia máxima atingida
    dataset_to_array[:,34], # frequencia cardíaca em repouso
    dataset_to_array[:,35], # pico de pressão sanguínea durante exercício 
    dataset_to_array[:,36], # pico de pressão sanguínea durante exercício  
    dataset_to_array[:,38] # pressão sanguínea em repouso
 ))

#dataset com os dados médicos do paciente
dataset_medico = np.column_stack((dataset_to_array[:,4] , # localização da dor
    dataset_to_array[:,6] , # alivio após o cansaço
    dataset_to_array[:,9] , # tipo de dor 
    dataset_to_array[:,39], # angina induzida pelo exercício (1 = sim; 0 = nao) 
    dataset.age, # idade 
    dataset.sex , # sexo
    dataset.hypertension # hipertensão
 ))

#concatena as duas bases de dados
dataset_paciente=np.concatenate((dataset_medico,dataset_sensor),axis=1)
print(dataset_paciente)

print(dataset_paciente.shape)

#encontrando os dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(dataset_paciente, target, random_state = 223)

#cria o objeto SVM
modelSVM = SVC(kernel = 'linear') #escolha do kernel polinomial

#aplica o treinamento ao modelo
modelSVM.fit(X_train, y_train)

"""**Analisando a performance do modelo**"""

previsao = modelSVM.predict(X_test) #aplica o modelo para os dados de teste

#encontra a acuracia do modelo de previsão utilizando o SVM 
accuracia = accuracy_score(y_test, previsao)
print ("Acuracia utilizando o SVM :" , accuracia , "\nEm porcentagem : ", round(accuracia*100) , "%\n")

#criando a matriz de confusão
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, previsao) #gera a matriz de confusão
df_cm = pd.DataFrame(cm, index = [i for i in "01234"],columns = [i for i in "01234"]) #cria o df com as classes
plt.figure(figsize = (10,7)) #indica o tamanho da figura 
sn.heatmap(df_cm, annot=True) #plota a figura

"""**Modificando o Dataset**"""

#vamos escolher apenas 13 atributos para realizar a previsão de doenças cardíacas

dataset_to_array = np.array(dataset)
label = dataset_to_array[:,57] # "Target" classes binárias 0 e 1
label = label.astype('int')
label[label>0] = 1 # Quando os dados são 0 está saudável e 1 doente

print(label)

#encontrando os dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(dataset_paciente, label, random_state = 223)

#cria o objeto SVM
modelSVM = SVC(kernel = 'linear') #escolha do kernel polinomial

#aplica o treinamento ao modelo
modelSVM.fit(X_train, y_train)

previsao = modelSVM.predict(X_test) #aplica o modelo para os dados de teste

#encontra a acuracia do modelo de previsão utilizando o SVM 
accuracia = accuracy_score(y_test, previsao)
print ("Acuracia utilizando o SVM :" , accuracia , "\nEm porcentagem : ", round(accuracia*100) , "%\n")

#criando a matriz de confusão
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, previsao) #gera a matriz de confusão
df_cm = pd.DataFrame(cm, index = [i for i in "01"],columns = [i for i in "01"]) #cria o df com as classes
plt.figure(figsize = (10,7)) #indica o tamanho da figura 
sn.heatmap(df_cm, annot=True) #plota a figura