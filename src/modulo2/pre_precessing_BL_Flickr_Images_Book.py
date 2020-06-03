"""
Created on Wed Jun  3 10:03:07 2020

@author: prbpedro
"""
import pandas
import numpy
import datetime

caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dataframe=pandas.read_csv(caminho_base + 'BL-Flickr-Images-Book.csv')

print(dataframe.head())
dataframe.info()

print(dataframe.isnull().sum())

#REMOÇÃO DE COLUNAS DESNECESSÁRIAS
columns_to_drop = ['Edition Statement', 'Corporate Contributors', 'Former owner',
                   'Engraver', 'Corporate Author', 'Contributors', 'Issuance type',
                   'Shelfmarks']
dataframe.drop(columns_to_drop, axis=1, inplace=True)
dataframe.info()
print(dataframe.isnull().sum())

# Verificação de existência de identificador no dataframe
print(dataframe['Identifier'].is_unique, dataframe['Identifier'].nunique(), sep=" - ")

# Transformação de dados despadronizados no formato yyyy
regex_year_four_digits = r'^(\d{4})'
pub_date_transformed = dataframe['Date of Publication'].str.extract(regex_year_four_digits, expand=False)
dataframe['Date of Publication'] = pandas.to_numeric(pub_date_transformed)

dataframe.info()
print(dataframe.isnull().sum())

# Retirada de linhas com valor nulo na coluna especificada
dataframe.dropna(subset=['Date of Publication'], inplace=True)

# Convert float para int
dataframe['Date of Publication'] = dataframe['Date of Publication'].astype(int)
dataframe.info()
print(dataframe.isnull().sum())

# Substituição dos valores da coluna que contém uma string específica
count_unique_pub_places = dataframe.groupby('Place of Publication').Identifier.nunique().sort_values(ascending=False)
contains_london_pub_place = dataframe['Place of Publication'].str.contains('London')
# dataframe.loc[dataframe['Place of Publication'].str.contains('London'), 'Place of Publication'] = 'London'
dataframe['Place of Publication'] = numpy.where(contains_london_pub_place, 'London', dataframe['Place of Publication'])
count_unique_pub_places = dataframe.groupby('Place of Publication').Identifier.nunique().sort_values(ascending=False)