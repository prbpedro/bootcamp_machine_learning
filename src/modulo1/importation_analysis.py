import numpy
import pandas
import seaborn
import matplotlib.pyplot
import google
import scipy.stats
import sklearn.metrics
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.tree
import sklearn.model_selection

def decisiontree_regression_analysis(dataframe):
    """Cria modelo de previsão por regressão de árvore de decisão linear para 
    colunas referenciadas, faz a análise desta e plota um gráfico de dispersão 
    dos dados previstos e da árvore de decisão"""

    # Obtem o dataframe reduzido
    reduced_dataframe = reduce_datraframe(dataframe)

    # Obtém o dataframe escalado de 0 a 1 de acordo com desvio padrão de cada registro
    # em relaçao ao mínimo e máximo valor da coluna a que este pertence
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_dataframe = scaler.fit_transform(reduced_dataframe)

    # Determina a entrada da árvore como os valores do índice 2 d dataframe referenciado
    # column declared_weight do dataframe original ou reduzido
    entrada_arvore=scaled_dataframe[:,2] 

    # Determina a entrada da árvore como os valores do índice 3 d dataframe referenciado
    # column actual_weight do dataframe original ou reduzido
    saida_arvore=scaled_dataframe[:,3] 

    # Transforma a entrada e saída em arrays de duas dimensões
    entrada_arvore_2d=entrada_arvore.reshape(-1,1)
    saida_arvore_2d=saida_arvore.reshape(-1,1)

    # Define a massa de teste e treinamento
    # test_size define a proporção x% para massa de teste e o restante para o treinamento
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        entrada_arvore_2d, saida_arvore_2d, test_size=0.30, random_state=42) 

    # Define a árvore de regressão e aplica a mesma aos dados de treinamento
    regression_tree=sklearn.tree.DecisionTreeRegressor() 
    regression_tree.fit(x_train, y_train) #aplica a regressão

    # Realiza a previsão com os dados de teste
    tree_prediction=regression_tree.predict(x_test)

    # Imprime a média do erro (valor real - valor predito) absoluto
    print('Média do erro absoluto:', sklearn.metrics.mean_absolute_error(y_test, tree_prediction))

    # Imprime a média do erro (valor real - valor predito) quadrático
    # Mede a qualidade do estimador (Não negativo)
    # Quanto mais próximo de 0 melhor o estimador
    print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, tree_prediction))

    # Plota gráfico da regressão com árvore de decisão 
    matplotlib.pyplot.figure(figsize=(15, 10))

    # Define os valores do eixo x
    X_grid = numpy.arange(min(entrada_arvore), max(entrada_arvore), 0.001)
    X_grid_2d = X_grid.reshape((len(X_grid), 1))

    # Realiza o plot do gráfico de dispersão
    matplotlib.pyplot.scatter(entrada_arvore, saida_arvore, color = 'red')

    # Realiza o plot da árvore de decisão
    matplotlib.pyplot.plot(X_grid_2d, regression_tree.predict(X_grid_2d), color = 'blue')

    matplotlib.pyplot.title('Exemplo de Regressão com Árvore de Decisão')
    matplotlib.pyplot.xlabel('declared_weight')
    matplotlib.pyplot.ylabel('actual_weight')
    matplotlib.pyplot.show()


def linearregression_analysis(dataframe):
    """Cria modelo de previsão por regressão linear para colunas referenciadas, 
    faz a análise desta e plota um gráfico de dispersão dos dados previstos e da 
    função linear de previsão"""

    linear_function = """Y = {}(A)*X + {}(B)
    Aonde a = Coeficiente Angular e b = Coeficiente Linear\n"""

    # Determina a variável independente 
    _x=dataframe['declared_weight'].values  

    # Determinavariável dependente 
    _y=dataframe['actual_weight'].values 

    # Obtem Formato 2D do array x
    x_2d=_x.reshape((-1, 1)) 

    # Calcula a regressão linear com base no array 2d criado a partir de x
    # e do array 1d criado a partir de y
    linear_regression=sklearn.linear_model.LinearRegression()
    linear_regression.fit(x_2d,_y) 

    # Realiza a previsão utilizando o array 2d criado a partir de x
    prediction=linear_regression.predict(x_2d)

    # Análise do modelo
    print(linear_function.format(linear_regression.coef_, linear_regression.intercept_))

    # Caulcula o coeficiente de determinação
    # Maio valor possível igual a 1, quanto maior melhor o encaixe
    # Indica se a previsão está de acordo com o real
    r2 = sklearn.metrics.r2_score(_y, prediction)  
    print("Coeficiente de Determinação (R2):", r2, end='\n\n')

    #realiza o plot dos dados
    matplotlib.pyplot.figure(figsize=(10, 10), dpi=100)

    # Realiza o plot do gráfico de dispersão
    matplotlib.pyplot.scatter(_x, _y,  color='gray') 

    # realiza o plot da função da linha de regressão
    matplotlib.pyplot.plot(_x, prediction, color='red', linewidth=2) 

    matplotlib.pyplot.xlabel("declared_weight")
    matplotlib.pyplot.ylabel("actual_weight")
    matplotlib.pyplot.show()

def linearregression_chinaitens_analysis(dataframe):
    """Cria modelo de previsão por regressão linear de amostra específica 
    dos dados do dataframe"""

    # Imprime os valores diferentes existentes para a coluna referenciada
    print(dataframe['country_of_origin'].unique())

    # Obtem dataframe somente com registros aonde 
    # o valor da coluna 'country_of_origin' é igual a 'China'
    dataframe_china_itens=dataframe[dataframe['country_of_origin']=='China']

    # Imprime os valores diferentes existentes para a coluna referenciada
    print(dataframe_china_itens['item'].nunique())

    linearregression_analysis(dataframe_china_itens)

def reduce_datraframe(dataframe):
    "Reduz o dataframe retirando as colunas referenciadas na função"
    return dataframe.drop(columns={
        "date_of_departure", 
        "date_of_arrival", 
        "date_of_departure", 
        "date_of_arrival", 
        "valid_import", 
        "item", 
        "importer_id", 
        "exporter_id", 
        "country_of_origin", 
        "mode_of_transport", 
        "route",
        "days_in_transit"
    }, inplace=False)

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

    reduced_dataframe = reduce_datraframe(dataframe)

    matplotlib.pyplot.figure(figsize=(20, 10))

    # Obtem a matrix de correlação
    corr_matrix = reduced_dataframe.corr()

    # Plota a matriz de correlação com o seaborn
    seaborn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,center= 0)  

    matplotlib.pyplot.show()


if __name__ == "__main__":
    # Cria o dataset do pandas a partir de csv
    dataframe = pandas.read_csv("src/input/Importation.csv")

    linearregression_analysis(dataframe)

    linearregression_chinaitens_analysis(dataframe)

    plot_correlation_matrix(dataframe)

    decisiontree_regression_analysis(dataframe)