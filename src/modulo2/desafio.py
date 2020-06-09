"""
Created on Tue Jun  9 16:49:25 2020

@author: prbpedro
"""

import pandas
import seaborn
import matplotlib
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder

def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import classification_report, confusion_matrix
    mc = confusion_matrix(y_test, y_pred)
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=mc)
    matplotlib.pyplot.show()

def executeKnn(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia KNN: ',acuracia)
    
def executeDecisionTree(X_train, X_test, y_train, y_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia DecisionTree: ',acuracia)
    
def executeRandomForest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia RandomForest: ',acuracia)
    
def executeSVM( X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    classifier = SVC(gamma='auto', kernel='rbf') # Kernel Linear
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia SVM: ',acuracia)

def executeMPL(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(alpha=1e-5,
                               hidden_layer_sizes=(5,5), random_state=1) 
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    from sklearn.metrics import accuracy_score
    acuracia = accuracy_score(y_test, y_pred)
    print('Acurácia MPL: ',acuracia)
    

def plot_correlation_matrix(dataframe):
    """ Plota gráfico de matriz de correlação das colunas do dataframe
    = –1 A perfect negative linear relationship
    < 0.70 A strong negative linear relationship
    < 0.50 A moderate negative relationship
    < 0.30 A weak negative linear relationship
    = 0 No linear relationship
    > 0.30 A weak positive linear relationship
    > 0.50 A moderate positive relationship
    > 0.70 A strong positive linear relationship
    = 1 A perfect positive linear relationship"""
    cs = []
    for c in dataframe.columns:
        if c!='fixed acidity' and c!='pH':
            cs.append(c)
    reduced_dataframe = dataframe.drop(columns=cs)
    #matplotlib.pyplot.figure(figsize=(20, 10))
    corr_matrix = reduced_dataframe.corr()
    seaborn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,center= 0)  
    matplotlib.pyplot.show()

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dataframe=pandas.read_csv(caminho_base + 'winequality-red.csv', sep=";")

#print(dataframe.head())
dataframe.info()

#verificando o tipo de dados
print(type(dataframe))  

#contando valores nulos 
#print(dataframe.isnull().sum())

for c in dataframe.columns:
    df_col = dataframe[c]
    print(df_col.describe(), end="\n\n")
    #verificando se existem outliers
    seaborn.boxplot(df_col)
    matplotlib.pyplot.show()

plot_correlation_matrix(dataframe)

df_reduced = dataframe[['alcohol','quality']]
matplotlib.pyplot.figure(figsize=(20, 10))
corr_matrix = df_reduced.corr()
seaborn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,center= 0)  
matplotlib.pyplot.show()

print(dataframe['quality'].value_counts())


dataframe["quality"] = dataframe["quality"].astype("category")
le = LabelEncoder()
le.fit(dataframe["quality"].values)
print(dataframe["quality"].cat.categories)
dataframe.info()

df_entrada = dataframe.drop(columns=["quality"])
df_saida = dataframe["quality"]

scaler = sklearn.preprocessing.MinMaxScaler()
scaled_dataframe = scaler.fit_transform(df_entrada)

print(scaled_dataframe[0].min())

x=scaled_dataframe
y=df_saida
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

executeKnn(x_train, x_test, y_train, y_test)
executeDecisionTree(x_train, x_test, y_train, y_test)
executeRandomForest(x_train, x_test, y_train, y_test)
executeSVM(x_train, x_test, y_train, y_test)
executeMPL(x_train, x_test, y_train, y_test)

dataframe["quality"] = dataframe["quality"].astype("float64")
dataframe['quality'][dataframe['quality']>5] = 99
dataframe['quality'][dataframe['quality']<=5] =0
print(dataframe['quality'].value_counts())
dataframe["quality"] = dataframe["quality"].astype("category")
le = LabelEncoder()
le.fit(dataframe["quality"].values)
print(dataframe["quality"].cat.categories)
dataframe.info()

df_saida = dataframe["quality"]
x=scaled_dataframe
y=df_saida
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

executeRandomForest(x_train, x_test, y_train, y_test)