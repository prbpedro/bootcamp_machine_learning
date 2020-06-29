"""
Created on Sun Jun 21 08:57:33 2020

@author: prbpedro
"""
import numpy
import pandas
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import seaborn
import matplotlib.pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

bunch = fetch_openml('mnist_784', version=1)

pd_dataframe = pandas.DataFrame(data=numpy.c_[bunch['data'], bunch['target']],
                                columns=bunch['feature_names'] + ['target'])

pd_dataframe.info()
pd_dataframe.isnull().any().describe()

seaborn.set()
seaborn.countplot(x="target", data=pd_dataframe)
matplotlib.pyplot.show()

pd_data_dataframe = pd_dataframe.loc[:, pd_dataframe.columns != 'target']
sample_digit = pd_data_dataframe.iloc[2000]
sample_digit_image = sample_digit.values.reshape(28, 28).astype((numpy.float))
matplotlib.pyplot.imshow(sample_digit_image,
                         cmap = matplotlib.cm.binary,
                         interpolation="nearest")
matplotlib.pyplot.title(pd_dataframe['target'].iloc[2000])
matplotlib.pyplot.axis("off")
matplotlib.pyplot.show()

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(bunch['data'])

X = scaled_X
y = bunch['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)

classifier_cross_val_score = cross_val_score(rf_clf, X, y)
print('Cros-validation: {0} & {1}'.format(classifier_cross_val_score.mean(), classifier_cross_val_score.std()))

print(classification_report(y_test, y_pred))

mc = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mc, figsize=(10, 10))
matplotlib.pyplot.title('Correlation Matrix')
matplotlib.pyplot.show()






param_range = numpy.arange(1, 250, 2)
train_scores, test_scores = validation_curve( rf_clf, X, y, param_range=param_range ,param_name="n_estimators", scoring="accuracy", n_jobs=1)
train_scores_mean = numpy.mean(train_scores, axis=1)
train_scores_std = numpy.std(train_scores, axis=1)
test_scores_mean = numpy.mean(test_scores, axis=1)
test_scores_std = numpy.std(test_scores, axis=1)

matplotlib.pyplot.title("Validation Curve with SVM")
matplotlib.pyplot.xlabel(r"$\gamma$")
matplotlib.pyplot.ylabel("Score")
matplotlib.pyplot.ylim(0.0, 1.1)
lw = 2
matplotlib.pyplot.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
matplotlib.pyplot.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
matplotlib.pyplot.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
matplotlib.pyplot.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
matplotlib.pyplot.legend(loc="best")
matplotlib.pyplot.show()