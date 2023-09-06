# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:29:07 2023

@author: User
"""

from sklearn.datasets import load_boston

data = load_boston()
print(data['DESCR'])
data['feature_names']

X, y = data['data'], data['target']
print("Размер матрицы объектов:", X.shape)
print("Размер вектора y:", y.shape)

from matplotlib import pyplot as plt
plt.scatter(X[:, 0], y)
plt.xlabel('Crime rate')
plt.ylabel('Price')

#separate data for train data and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 5, weights='uniform', p=2)

knn.fit(X_train, y_train)

knn.predict(X_test)

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, knn.predict(X_test))

#Choice giperparametrs

from sklearn.model_selection import GridSearchCV
from sklearn import KFold

grid_searcher = GridSearchCV(KNeighborsRegressor(),
                             param_grid={'n_neighbors': [1, 5 ,10, 20],
                                         'weights': ['uniform', 'distance'],
                                         'p': [1, 2, 3]},
                             cv=5)

grid_searcher.fit(X_train, y_train)
grid_searcher.best_params_
grid_searcher.predict(X_test)

#calculate metric
mean_squared_error(y_test, grid_searcher.predict(X_test))

metrics = []

for n in range (1, 30, 3):
    knn = KNeighborsRegressor(n_neighbors=n)
    knn.fit(X_train, y_train)
    metrics.append(mean_squared_error(y_test, knn.predict(X_test)))

plt.plot(range(1, 30, 3), metrics)
plt.ylabel('Negative mean squared error')
plt.xlabel('Number of neightbors')

#classification
from sklearn.datasets import make_moons
X, y = make_moons()
X.shape
plt.scatter(X[:, 0], X[:, 1], c=y)
X, y = make_moons(n_samples=1000, noise=0.25)
plt.scatter(X[:, 0], X[:, 1], c=y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn.fit(X_train, y_train)
knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, knn.predict(X_test))

import numpy as np
x_space = np.linspace(-2, 2, 100)
x_grid, y_grid = np.meshgrid(x_space, x_space)
x_grid
y_grid
xy = np.stack([x_grid, y_grid], axis=2).reshape(-1, 2)
xy

plt.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.4, c=knn.predict(xy))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

#Homework
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

X, y = make_moons(n_samples=1000, noise=0.5, random_state=10)
plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

grid_searcher1 = GridSearchCV(KNeighborsClassifier(),
                              param_grid = {'n_neighbors' : [i for i in range(1, 21)],
                                            'weights' : ['uniform', 'distance'],
                                            'p' : [1, 2 ,3]},
                              cv = KFold(n_splits=5, random_state=10, shuffle=True))
grid_searcher1.fit(X_train, y_train)
grid_searcher1.best_params_
grid_searcher1.predict(X_test)
mean_squared_error(y_test, grid_searcher1.predict(X_test))
metrics = []
for n in [i for i in range(1, 21)]:
    knn1 = KNeighborsRegressor(n_neighbors = n)
    scores = cross_val_score(knn1, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    metrics.append(np.mean(scores))
plt.plot([i for i in range(1, 21)], metrics)
plt.ylabel('Negative mean squared error')
plt.xlabel('Number of neightbors')


def train_grid_search(X, y):
    def_searcher = GridSearchCV(KNeighborsClassifier(), param_grid = {'n_neighbors' : [i for i in range(1,21)]})
    def_searcher.fit(X_train,y_train)
    return def_searcher.cv_results_['mean_test_score']

mean_test_scores = []
for i in range(1000):
    X, y = make_moons(n_samples = 1000, noise = 0.5)
    mean_test_score = train_grid_search(X, y)
    mean_test_scores.append(mean_test_score)

mean_test_scores = np.array(mean_test_scores)
plt.plot(np.arange(1, 21), np.mean(mean_test_scores, axis = 0))

