"""
Created on Mon Jul 13 12:27:33 2020

@author: prbpedro
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
    
caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
dataframe=pd.read_csv(caminho_base + 'cars.csv')

dataframe.info()


dataframe['cubicinches'] = pd.to_numeric(dataframe['cubicinches'], errors='coerce')
dataframe['weightlbs'] = pd.to_numeric(dataframe['weightlbs'], errors='coerce')

print(dataframe[dataframe['cubicinches'].isnull()].index.tolist())
print(dataframe[dataframe['weightlbs'].isnull()].index.tolist())


print(dataframe.isnull().sum())



dataframe['cubicinches'] = dataframe['cubicinches'].fillna((dataframe['cubicinches'].mean()))
dataframe['cubicinches'] = dataframe['cubicinches'].astype('int64')


dataframe['weightlbs'] = dataframe['weightlbs'].fillna((dataframe['weightlbs'].mean()))
dataframe['weightlbs'] = dataframe['weightlbs'].astype('int64')

dataframe['year'] = dataframe['year'].astype('int64')

print(dataframe['weightlbs'].mean())
dataframe.info()

for c in  ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']:
    print(dataframe[c].describe(), end="\n\n")  

print(dataframe['brand'].unique())

dataframe['brand'] = dataframe['brand'].astype('category')
dataframe.info()

for columnname in dataframe:
    if columnname not in ('brand'):
        plt.figure(figsize=(10, 10))
        g = sns.boxplot(dataframe[columnname]) 
        g.set_title('Box plot ' + columnname)
        
clean_df = dataframe.drop(['brand'], axis=1)

correlation = dataframe.corr()
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='viridis')
plt.title('Correlation between different fearures')
plt.show();


dataframe['eficiencia'] = dataframe['mpg'] > 25

dataframe.info()

X_std = StandardScaler().fit_transform(clean_df)

pca = PCA(n_components=7)
principalComponents = pca.fit_transform(X_std)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
PCA_components = PCA_components.drop(columns=[0,6], axis=1)


kmeans = KMeans(n_clusters=3, random_state=42) 
y_kmeans = kmeans.fit_predict(PCA_components)

X_PCA = PCA_components.values

plt.scatter(X_PCA[y_kmeans == 0, 0], X_PCA[y_kmeans == 0,1],s=100,c='red',label='US')
plt.scatter(X_PCA[y_kmeans == 1, 0], X_PCA[y_kmeans == 1,1],s=100,c='blue',label='Japan')
plt.scatter(X_PCA[y_kmeans == 2, 0], X_PCA[y_kmeans == 2,1],s=100,c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.title('Clusters of car brands')
plt.legend()
plt.show()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test=train_test_split(clean_df, dataframe['brand'], test_size=0.2)

# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state=42)
# classifier.fit(X_train, y_train)
# y_pred = classifier.predict(X_test)

# from sklearn.metrics import classification_report, confusion_matrix
# mc = confusion_matrix(y_test, y_pred)
# print(mc)
# print(classification_report(y_test, y_pred))

# from mlxtend.plotting import plot_confusion_matrix
# fig, ax = plot_confusion_matrix(conf_mat=mc)
# plt.show()



X_train, X_test, y_train, y_test=train_test_split(PCA_components, dataframe['eficiencia'].values, test_size=0.3)
print('LogisticRegression')
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
mc = confusion_matrix(y_test, y_pred)
print(mc)
print(classification_report(y_test, y_pred))
fig, ax = plot_confusion_matrix(conf_mat=mc)
plt.show()
print('')
print('DecisionTreeClassifier')
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
mc = confusion_matrix(y_test, y_pred)
print(mc)
print(classification_report(y_test, y_pred))
fig, ax = plot_confusion_matrix(conf_mat=mc)
plt.show()

principalComponents[:,3].max()