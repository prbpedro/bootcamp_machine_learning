"""
Created on Wed Jun  3 10:03:07 2020

@author: prbpedro
"""
import pandas
import matplotlib.pyplot;
from sklearn.linear_model import LinearRegression, BayesianRidge, LassoLars, OrthogonalMatchingPursuit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import seaborn

if __name__ == "__main__":
    
    caminho_base='/home/prbpedro/Development/repositories/github/bootcamp_artificial_intelligence/src/input/'
    dataframe=pandas.read_csv(caminho_base + 'Importation.csv')
    
    dataframe.info()

    X=dataframe[['declared_weight', 'declared_quantity', 'declared_cost']]

    y=dataframe['actual_weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    corrdataframe = dataframe[['declared_quantity', 'declared_cost','declared_weight', 'actual_weight']]
    corr_matrix = corrdataframe.corr()
    seaborn.heatmap(corr_matrix, annot=True,vmin=-1, vmax=1,center= 0)
    matplotlib.pyplot.show()
    
    corrdataframe.plot(kind='hist')
    matplotlib.pyplot.show()
        
    # Classificadores comparados
    classifiers = ((LinearRegression(),None),
                   (BayesianRidge(), {'n_iter':(1,500)}),
                   (LassoLars(), {'alpha': (1,10)}), 
                   (OrthogonalMatchingPursuit(), None),
                   (DecisionTreeRegressor(), {'min_samples_split': (2,20),
                                              'max_depth': (1,10)}),)
    
    for classifier, parameters in classifiers:
        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        
        classifier_name = type(classifier).__name__
        print('-------------------------------- {0} --------------------------------'.format(classifier_name))
        
        print('mean_squared_error %.2f' % mean_squared_error(y_test, y_predicted))
        print('median_absolute_error: %.2f' % median_absolute_error(y_test, y_predicted))
        print('mean_absolute_error: %.2f' % mean_absolute_error(y_test, y_predicted))
        print('r2_score: %.2f' % r2_score(y_test, y_predicted))
        
        
        if parameters != None:
            print('\nParameters evaluation:')
            from sklearn.model_selection import GridSearchCV
            parameters_classifier = GridSearchCV(classifier, parameters)
            parameters_classifier.fit(X, y)
            for k, v in parameters_classifier.best_params_.items():
                print('Best value for {0}: {1}'.format(k, v))
        print('---------------------------------------------------------------------------')