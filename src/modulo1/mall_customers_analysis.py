import numpy
import pandas
import seaborn
import matplotlib.pyplot
import google
import scipy.stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def addnulls(dataframe):
    """Método que adiciona valores nulos de posição aleatória no dataframe """
    for c in dataframe.columns:
        dataframe.loc[dataframe.sample(frac=0.1).index, c] = numpy.nan

def removenulls(dataframe, drop = False):
    """Método que preenche valores nulos no dataframe com a média dos valores
    da coluna a que pertence o valor nulo ou deleta os resgistros 
    com valor nulo do dataframe.
    """
    if(drop):
        # Deleta do dataframe registros com algum valor nulo
        return dataframe.dropna()

    #Altera os valores nulos com a média da coluna do valor
    return dataframe.fillna(dataframe.mean())

def verify_outleirs(dataframe):
    """Método de auxílio para verificação de outleirs através de captura de dados de 
    amostra de uma coluna aonde o zscore absoluto dos valores desta são maiores que
    um threshold determinado. Também faz o BoxPlot de colunas específicas do dataframe.
    """
    
    # Retorna o array de zcore absoluto referente a coluna referenciada do data set
    # 1-68% 2-95% 3-99.7% (% dados dentro da amosta)
    z = numpy.abs(scipy.stats.zscore(dataframe["Annual Income (k$)"].values))

    # 2-95%
    threshold = 2

    # Retorna a amostra de resultados aonde o zscore de cada elemento é testado
    # com a codição de ser maior que o threshold referenciado Tupla(index, true|false)
    result = numpy.where(z > threshold)

    # Retorna a amostra de dados relativas ao número dos índices referenciados
    # outleirs
    ds_outliers = dataframe.iloc[result[0]]
    print(ds_outliers, end="\n\n")

    # BoxPlot matplotlib - dataframe
    boxplot = dataframe.boxplot(column = ["Age", "Annual Income (k$)", "Spending Score (1-100)"])
    matplotlib.pyplot.show()

def print_countplot(dataframe):
    """Plota gráfico de distriuição de dados com a contagem determinada pelos valores diferentes
    de uma determinada coluna do dataframe"""

    seaborn.countplot(x="Gender", data=dataframe)
    matplotlib.pyplot.title("Gender distribution")
    matplotlib.pyplot.show()

def print_histogram(dataframe):
    """Plota gráfico historiograma 
    Contagem de registros a paritr de valores iguais de uma determinada coluna)"""

    # Determina o historiagorama a partir da coluna referenciada
    # com agrupamento (valores do Eixo y) determinado pela variável bins
    dataframe.hist("Age", bins=35)

    matplotlib.pyplot.title("Customers distribution by age")
    matplotlib.pyplot.xlabel("Age")
    matplotlib.pyplot.show()

def codificarliterais(dataframe, encoder = None):
    """Método que identica os tipos object no dataframe e transforma a coluna
    Gender para a repreentação de valores inteiros ao invés de literais"""

    if(encoder and encoder == 'LabelEncoder'): 
        # Cria o LabelEncoder capaz de transformar valores de uma dada coluna (label)
        # Em valores numéricos categorizados
        le = LabelEncoder()
        dataframe["Gender"] = le.fit_transform(dataframe["Gender"])
        return
    
    if(encoder and encoder == 'OneHotEncoder'): 
        # Cria o OneHotEncoder capaz de transformar valores de uma dada coluna
        # Em valores numéricos categorizados
        ohe = OneHotEncoder()
        dataframe["Gender"] = ohe.fit_transform(dataframe["Gender"].values.reshape(-1,1)).toarray()
        return

    # Retorna um dataframe que contém somente as colunas do tipo object presentes
    # no dataframe original
    cat_dataframe = dataframe.select_dtypes(include=["object"])

    # Retorna uma Serie do tipo Categorical para a coluna referenciada
    gendercategoricalserie = cat_dataframe["Gender"].astype("category")

    # Retorna uma lista com as possíveis categorias da Serie referenciada
    labels = gendercategoricalserie.cat.categories.tolist()

    # Cria um objeto que especifica os valores numéricos para cada categoria retornada no passo anterior
    # sendo o valor desta igual ao seu índice no array labels + 1
    replace_map_comp = {"Gender": {k: v for k,v in zip(labels, list(range(1, len(labels)+1)))}}

    # Substitui os valores dos registros conforme especificados pelo objeto de parâmetro
    dataframe.replace(replace_map_comp, inplace=True)


if __name__ == "__main__":

    # Cria o dataframe do pandas a partir de csv
    dataframe = pandas.read_csv("src/input/MallCustomers.csv")

    # Retorna os primeiros 5 elementos do dataframe
    print(dataframe.head(), end="\n\n")

    # Retorna a quantidade total de registros e informações sobre as colunas 
    # presentes no dataframe
    print(dataframe.info(), end="\n\n")

    # Retorna informações calculadas count, mean, std, min, 25%, 50%, 75% e max 
    # referente a cada coluna numérica do dataframe
    print(dataframe.describe(), end="\n\n")

    # Retorna a soma dos registros com valor nulo em cada uma das colunas 
    # do dataframe
    print(dataframe.isnull().sum(), end="\n\n")

    codificarliterais(dataframe, 'OneHotEncoder')

    addnulls(dataframe)

    dataframe = removenulls(dataframe, True)
    
    verify_outleirs(dataframe)

    print_countplot(dataframe)

    print_histogram(dataframe)
