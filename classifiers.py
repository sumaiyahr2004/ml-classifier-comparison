from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        self.training_data, self.testing_data, self.training_labels, self.testing_labels = \
            train_test_split(X, y, test_size=0.4, random_state=42)

        self.outputs = []

        plt.figure()
        for label in set(y):
            plt.scatter(X[y == label, 0], X[y == label, 1], label=f'Class {label}')
        plt.legend()
        plt.title('Dataset Scatter Plot')
        plt.xlabel('Feature A')
        plt.ylabel('Feature B')
        plt.savefig('scatter_plot.png')
        plt.close()

    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        clf.fit(self.training_data, self.training_labels)
        best_model = clf.best_estimator_
        train_score = clf.best_score_
        test_score = best_model.score(self.testing_data, self.testing_labels)
        print(f'  Best params : {clf.best_params_}')
        print(f'  Train acc   : {train_score:.4f}')
        print(f'  Test  acc   : {test_score:.4f}')
        self.outputs.append(f'{classifier_name}, {train_score:.4f}, {test_score:.4f}')
        self.plot(self.testing_data, best_model.predict(self.testing_data), model=best_model, classifier_name=classifier_name)

    def classifyNearestNeighbors(self):
        # TODO: Write code to run a Nearest Neighbors classifier
        param_grid = {
            'n_neighbors': list(range(1, 20, 2)),
            'leaf_size':   list(range(5, 35, 5)),
        }
        clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=1)
        self.test_clf(clf, classifier_name='KNN')

    def classifyLogisticRegression(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
        clf = GridSearchCV(LogisticRegression(max_iter=1000, solver='lbfgs'),
                           param_grid, cv=5, scoring='accuracy', n_jobs=1)
        self.test_clf(clf, classifier_name='Logistic Regression')

    def classifyDecisionTree(self):
        # TODO: Write code to run a Logistic Regression classifier
        param_grid = {
            'max_depth':         list(range(1, 51)),
            'min_samples_split': list(range(2, 11)),
        }
        clf = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=1)
        self.test_clf(clf, classifier_name='Decision Tree')

    def classifyRandomForest(self):
        # TODO: Write code to run a Random Forest classifier
        param_grid = {
            'max_depth':         list(range(1, 6)),
            'min_samples_split': list(range(2, 11)),
        }
        clf = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=1)
        self.test_clf(clf, classifier_name='Random Forest')

    def classifyAdaBoost(self):
        # TODO: Write code to run a AdaBoost classifier
        param_grid = {'n_estimators': list(range(10, 80, 10))}
        clf = GridSearchCV(AdaBoostClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=1)
        self.test_clf(clf, classifier_name='AdaBoost')

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]
        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5
        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.
        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)
        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)
        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)
        plt.margins(0.0)
        # reminder to uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')
        plt.show()

if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()
    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)
