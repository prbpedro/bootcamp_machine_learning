"""
Created on Wed Jun  3 10:03:07 2020

@author: prbpedro
"""
import pandas
import numpy
import matplotlib.pyplot

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dataframe=pandas.read_csv(caminho_base + 'pima-indians-diabetes.csv', header=None)

# Definindo o header
dataframe.columns=['Quantidade de vezes grávida', 'Concentração de glicose no plasma',
                   'Pressão diastólica (mm Hg)',
                   'Triceps skinfold thicknerss (mm)', 'Insulina em jejum (mm U/ml)',
                   'Índice de massa corporal (peso em kg/(altura em metros)^2',
                   'Diabetes predigree function', 'Idade', 'Possui diabes']
dataframe.info()

# Transformação do tipo de coluna
dataframe['Possui diabes'] = dataframe['Possui diabes'].astype('bool')
dataframe.info()

print(dataframe.describe())

# Criação de nova coluna baseada em valores de outra coluna
lambda_f = lambda row: row[4]>120 if True else False
dataframe['Glicose Alta'] = dataframe.apply(lambda_f, axis=1)
dataframe.info()
dataframe.describe()

# Obtem os itens com valor igual a 0 em alguma das colunas
cols = ['Concentração de glicose no plasma',
        'Pressão diastólica (mm Hg)',
        'Triceps skinfold thicknerss (mm)', 'Insulina em jejum (mm U/ml)', 'Idade']
dataframe_bad_zeros=dataframe[dataframe.eq(0).any(1)]
print((dataframe[cols]==0).sum())

# Substituição de valores irreais iguais a 0 por NaN
dataframe[cols] = dataframe[cols].replace(0,numpy.NaN)
print(dataframe.isnull().sum())

# Remoção de nulos por DROP
df_no_null = dataframe.dropna()
df_no_null.info()

# Substituição de nulos pela média da coluna
d = dataframe.mean()
df_no_null_mean = dataframe.fillna(dataframe.mean())
df_no_null_mean.info()

# Verificação de outliers
boxplot = dataframe.boxplot(column = cols)
matplotlib.pyplot.show()
