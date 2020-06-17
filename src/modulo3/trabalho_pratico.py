"""
Created on Wed Jun 17 12:01:38 2020

@author: prbpedro
"""

from sklearn import datasets
import numpy
import pandas
import matplotlib.pyplot

sklearn_dataset = datasets.load_iris()

pd_dataframe = pandas.DataFrame(data=numpy.c_[sklearn_dataset['data'], sklearn_dataset['target']],
                     columns=sklearn_dataset['feature_names'] + ['target'])

def pre_process(sklearn_dataset, pd_dataframe):
    pd_dataframe.info()
    
    # Transformação do target em tipo categórico
    print(pd_dataframe['target'].value_counts())
    pd_dataframe['target'] = pd_dataframe['target'].astype('category')
    
    # Sem nulos
    print(pd_dataframe.isnull().sum())
    
    pd_dataframe.info()
    
    # Sem outliers
    pd_dataframe.boxplot(figsize=(8,8), column = sklearn_dataset['feature_names'])
    matplotlib.pyplot.show()
    
    print(pd_dataframe.describe())
    
    from sklearn.preprocessing import label_binarize
    X = pd_dataframe.iloc[:,0:4]
    y = label_binarize(pd_dataframe['target'], classes=numpy.unique(pd_dataframe['target']))
    
    return X, y
    
def plot_confusion_matrix(classifier_name, classifier, X_test, y_test, y_pred):
    from sklearn.metrics import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix
    mc = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    fig, ax = plot_confusion_matrix(conf_mat=mc)
    matplotlib.pyplot.title('{0} - Correlation Matrix'.format(classifier_name))
    matplotlib.pyplot.show()
    
def plot_roc_curve(classifier_name, n_classes, y_test, y_score):
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        matplotlib.pyplot.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format( i, roc_auc[i]))
    
    matplotlib.pyplot.plot([0, 1], [0, 1], 'k--')
    matplotlib.pyplot.xlim([-0.05, 1.0])
    matplotlib.pyplot.ylim([0.0, 1.05])
    matplotlib.pyplot.xlabel('False Positive Rate')
    matplotlib.pyplot.ylabel('True Positive Rate')
    matplotlib.pyplot.title('{0} - Multiclass roc curve'.format(classifier_name))
    matplotlib.pyplot.legend(loc="lower right")
    matplotlib.pyplot.show()
        
X, y = pre_process(sklearn_dataset, pd_dataframe);

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

# Classificadores comparados
classifiers = ((KNeighborsClassifier(n_neighbors=5), {'n_neighbors':(1,20),
                                                      'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}),
               (RandomForestClassifier(n_estimators=100, random_state=42), {'min_samples_split':(2,100),
                                                                            'n_estimators': (5, 300)}),
               (DecisionTreeClassifier(random_state=42), {'min_samples_split': (2,50),
                                                          'min_samples_leaf': (1,50)}), 
               (SVC(gamma='auto', probability=True, random_state=42), {'kernel': ('linear', 'rbf', 'poly')}), 
               (MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5,5), random_state=42), None),
               (LogisticRegression(solver='lbfgs'), None),
               (GaussianNB(),None),)

for classifier_alg, parameters in classifiers:
    classifier = OneVsRestClassifier(classifier_alg)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_scores = classifier.predict_proba(X_test)
    
    classifier_name = type(classifier_alg).__name__
    print('-------------------------------- {0} --------------------------------'.format(classifier_name))
    from sklearn.model_selection import cross_val_score
    classifier_cross_val_score = cross_val_score(classifier, X, y)
    print('Cros-validation: {0} & {1}'.format(classifier_cross_val_score.mean(), classifier_cross_val_score.std()))
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(classifier_name, classifier_alg, X_test, y_test, y_pred)
    plot_roc_curve(classifier_name, y.shape[1], y_test, y_scores)
    
    if parameters != None:
        print('Parameters evaluation:')
        from sklearn.model_selection import GridSearchCV
        gs_X = pd_dataframe.iloc[:,0:4]
        gs_y = pd_dataframe['target']
        parameters_classifier = GridSearchCV(classifier_alg, parameters)
        parameters_classifier.fit(gs_X, gs_y)
        for k, v in parameters_classifier.best_params_.items():
            print('Best value for {0}: {1}'.format(k, v))
    print('---------------------------------------------------------------------------')

