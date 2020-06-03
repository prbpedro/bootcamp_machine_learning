"""
Created on Wed Jun  3 10:03:07 2020

@author: prbpedro
"""
import pandas

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dataframe=pandas.read_csv(caminho_base +   'imports-85.txt', sep=",", header=None, na_values="?") 

# Renomear os nomes das colunas
headers = {0:"symboling", 1:"normalized_losses", 2:"make", 3:"fuel_type", 4:"aspiration",
           5:"num_doors", 6:"body_style", 7:"drive_wheels",8: "engine_location",
           9:"wheel_base", 10:"length", 11:"width",12: "height", 13:"curb_weight",
           14:"engine_type", 15:"num_cylinders",16: "engine_size", 17:"fuel_system",
           18:"bore", 19:"stroke", 20:"compression_ratio", 21:"horsepower", 22:"peak_rpm",
           23:"city_mpg", 24:"highway_mpg", 25:"price"}
dataframe.rename(columns=headers,inplace=True)
dataframe.info()

# conta a quantidade de "object" existente no dataset
print(dataframe.dtypes.eq('object').sum())

#selecionando apenas os dados categóricos
#realiza uma cópia do dataset apenas para as colunas do tipo string
dataframe_string = dataframe.select_dtypes(include=['object']).copy()
dataframe_string.head()
dataframe_string.info()

#encontra onde existem valores nulos
print(dataframe_string.isnull().sum())

#realiza o drop para os valroes nulos
dataframe_string.dropna(inplace=True) 
dataframe_string.info()

#conta os valores para cada uma das "classes"
print(dataframe_string["num_cylinders"].value_counts())
print(dataframe_string["num_doors"].value_counts())

#utilizando o dicionário e a função "replace" para transformar em numericos
mapeamento_classes = {"num_doors":     {"four": 4, "two": 2},
                "num_cylinders": {"four": 4, "six": 6, "five": 5, "eight": 8,
                                  "two": 2, "twelve": 12, "three":3 }}
dataframe_string.replace(mapeamento_classes, inplace=True)

#aplicando o label encoding
#define a coluna de 'body_style' como categórica
dataframe_string["body_style"] = dataframe_string["body_style"].astype('category')  
print(dataframe_string.dtypes)

#após definir como categórica é possível realizar o procedimento de label encoding
#aplica a codificação para a coluna "body_style" e cria a coluna "body_style_cat"
dataframe_string.info()
dataframe_string["body_style_cat"] = dataframe_string["body_style"].cat.codes 
dataframe_string["body_style_cat"] = dataframe_string["body_style_cat"].astype('category')  
dataframe_string.info()

#aplicando o One-Hot_encoding
print(dataframe_string['drive_wheels'].unique())

#aplicando o hot encoding sobre os valores tração dos carros
drive_wheels_dummies = pandas.get_dummies(dataframe_string, columns=["drive_wheels"])