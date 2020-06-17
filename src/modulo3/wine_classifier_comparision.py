import matplotlib.pyplot
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

dataset = datasets.load_wine()
X = dataset.data[:, :]
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regression_classifier = RandomForestClassifier()
regression_classifier.fit(X_train, y_train)
y_predicted = regression_classifier.predict(X_test)

regression_classifier_accuracy_score = round(accuracy_score(y_test, y_predicted),6)
regression_classifier_recall_score = round(recall_score(y_test, y_predicted, average="weighted"),6)
regression_classifier_precision_score = round(precision_score(y_test, y_predicted, average="weighted"),6)
regression_classifier_cross_val_score = cross_val_score(regression_classifier, X, y)
sum_regression_classifier_cross_val_score = 0
for cv in regression_classifier_cross_val_score:
    sum_regression_classifier_cross_val_score += cv
mean_regression_classifier_cross_val_score = sum_regression_classifier_cross_val_score/len(regression_classifier_cross_val_score)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
y_predicted = knn_classifier.predict(X_test)

knn_classifier_accuracy_score = round(accuracy_score(y_test, y_predicted),6)
knn_classifier_recall_score = round(recall_score(y_test, y_predicted, average="weighted"),6)
knn_classifier_precision_score = round(precision_score(y_test, y_predicted, average="weighted"),6)
knn_classifier_cross_val_score = cross_val_score(knn_classifier, X, y)
sum_knn_classifier_cross_val_score = 0
for cv in knn_classifier_cross_val_score:
    sum_knn_classifier_cross_val_score += cv
mean_knn_classifier_cross_val_score = sum_knn_classifier_cross_val_score/len(knn_classifier_cross_val_score)

print('Random Forest vs Knn')
print('Classes: {0}'.format(dataset.target_names))
print('Acurácia: {0} vs {1}'.format(regression_classifier_accuracy_score, knn_classifier_accuracy_score))
print('Recall: {0} vs {1}'.format(regression_classifier_recall_score, knn_classifier_recall_score))
print('Precisão: {0} vs {1}'.format(regression_classifier_precision_score, knn_classifier_precision_score))
print('Validaçã cruzada: {0} vs {1}'.format(regression_classifier_cross_val_score, knn_classifier_cross_val_score))
print('Média das validações cruzadas: {0} vs {1}'.format(mean_regression_classifier_cross_val_score, mean_knn_classifier_cross_val_score))

parameters = {'min_samples_split':(2,6)}
hps_regression_classifier = GridSearchCV(regression_classifier, parameters)
hps_regression_classifier.fit(X, y)
print('Melhor valor para min_samples_split: {0}'.format(hps_regression_classifier.best_params_['min_samples_split']))

parameters = {'n_neighbors':(1,20)}
hps_knn_classifier = GridSearchCV(knn_classifier, parameters)
hps_knn_classifier.fit(X, y)
print('Melhor valor para n_neighbors: {0}'.format(hps_knn_classifier.best_params_['n_neighbors']))