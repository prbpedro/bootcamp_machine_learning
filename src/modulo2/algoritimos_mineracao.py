"""
Created on Mon Jun  8 15:57:44 2020

@author: prbpedro
"""

import pandas
import matplotlib.pyplot
import numpy

def executeKMeans():
    """
    Método de Clustering que objetiva particionar n observações dentre k grupos 
    onde cada observação pertence ao grupo mais próximo da média. Isso resulta 
    em uma divisão do espaço de dados em um Diagrama de Voronoi. 
    """
    from random import seed
    from random import randint
    
    seed(1)
    data = {'x': [randint(0,99) for _ in range(30)],
            'y': [randint(0,99) for _ in range(30)]
            }
    print(data)
    
    df = pandas.DataFrame(data, columns=['x', 'y'])
    df.info()

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df)
    centroids = kmeans.cluster_centers_ #Coordenadas dos centróids
    print(centroids)
    
    matplotlib.pyplot.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    matplotlib.pyplot.scatter(centroids[:,0], centroids[:,1], c='red', s=50)
    matplotlib.pyplot.xlabel('X')
    matplotlib.pyplot.xlabel('Y')

def read_iris_dataset():
    from sklearn import datasets
    iris_ds = datasets.load_iris();
    df_iris = pandas.DataFrame(data=numpy.c_[iris_ds['data'], iris_ds['target']],
                               columns=iris_ds['feature_names'] + ['target'])
    df_iris.info()
    
    X = df_iris.iloc[:,:-1] # dados de entrada (features) (tudo menos target)
    y = df_iris.iloc[:,4] # dados de saída (target)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)
    
    # Standardize features by removing the mean and scaling to unit variance
    # Média do desvio padrão
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return  iris_ds, df_iris, X_train, X_test, y_train, y_test

def executeKnn(iris_ds, df_iris, X_train, X_test, y_train, y_test):
    """
    Algoritimo supervisionado que determina o rótulo de classificação de uma 
    amostra baseado nas amostras vizinhas advindas de um conjunto de treinamento
    """
    
    
    # Treinamento do modeloiris_ds, df_iris, X_train, X_test, y_train, y_test
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    
    # Previsão
    y_pred = classifier.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred)
    
def plot_confusion_matrix(y_test, y_pred):
    # confusion_matrix e classification_report
    from sklearn.metrics import classification_report, confusion_matrix
    mc = confusion_matrix(y_test, y_pred)
    print(mc)
    print(classification_report(y_test, y_pred))
    
    from mlxtend.plotting import plot_confusion_matrix
    fig, ax = plot_confusion_matrix(conf_mat=mc)
    matplotlib.pyplot.show()
    
def executeDecisionTree(iris_ds, df_iris, X_train, X_test, y_train, y_test):
    """
    Algoritimo supervisionado de classificação por árvore de decisão.
    """
    from sklearn.tree import DecisionTreeClassifier
    
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    
    # Previsão
    y_pred = classifier.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred)
    plot_arvore_decisao(classifier, iris_ds['feature_names'])
    plot_arvore_decisao
    
def plot_arvore_decisao(classifier, f_names):
    from sklearn.tree import export_graphviz
    from IPython.display import Image
    import pydotplus
    dot_data = export_graphviz(classifier, out_file=None, filled=True,
                               rounded=True, special_characters=True, 
                               feature_names = f_names, class_names=['0','1','2'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('iris.png')
    Image(graph.create_png())

def executeSVM(iris_ds, df_iris, X_train, X_test, y_train, y_test):
    """
    Algoritimo supervisionado de classificação por SVM.
    Hiperplano de separação das classes.
    """
    # Treinamento do modelo
    from sklearn.svm import SVC
    classifier = SVC(gamma='auto') # Kernel Linear
    classifier.fit(X_train, y_train)
    
    # Previsão
    y_pred = classifier.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred)

def executeMPL(iris_ds, df_iris, X_train, X_test, y_train, y_test):
    """
    Algoritimo supervisionado de classificação por rede neural covolucional.
    Hiperplano de separação das classes.
    """
    # Treinamento do modelo
    from sklearn.neural_network import MLPClassifier
    # rede com duas camadas escondidas com 5 neuronios cada
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,
                               hidden_layer_sizes=(5,5), random_state=1) 
    classifier.fit(X_train, y_train)
    
    # Previsão
    y_pred = classifier.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred)
    
if __name__ == '__main__':
    iris_ds, df_iris, X_train, X_test, y_train, y_test=read_iris_dataset()
    executeKMeans()
    executeKnn(iris_ds, df_iris, X_train, X_test, y_train, y_test)
    executeDecisionTree(iris_ds, df_iris, X_train, X_test, y_train, y_test)
    executeSVM(iris_ds, df_iris, X_train, X_test, y_train, y_test)
    executeMPL(iris_ds, df_iris, X_train, X_test, y_train, y_test)