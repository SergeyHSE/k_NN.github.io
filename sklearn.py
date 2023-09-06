# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 00:29:07 2023

@author: SergeyHSE
"""
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

"""
Firstly let's generate dataset, split it on train and test and iterate through the parametrs
"""

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

