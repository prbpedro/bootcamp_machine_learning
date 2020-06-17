import matplotlib.pyplot
import numpy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split

X, y = datasets.load_diabetes(return_X_y=True)

# Somente uma feature, a do índice 2
X = X[:, numpy.newaxis, 2]

# Divide a massa de teste e treinamento

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Treinamento da regressão Linear
regression = linear_model.LinearRegression()
regression.fit(X_train, y_train)

# Realiza a previsão com dados de teste
y_predicted = regression.predict(X_test)

print('Erro médio quadrático: %.2f' % mean_squared_error(y_test, y_predicted))
print('Erro mediano absoluto: %.2f' % median_absolute_error(y_test, y_predicted))

matplotlib.pyplot.scatter(X_test, y_test, color='black')
matplotlib.pyplot.plot(X_test, y_predicted, color='blue')
matplotlib.pyplot.show()