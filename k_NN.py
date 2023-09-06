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

plt.figure(figsize=(9, 9), dpi=100)
ax = plt.gca()
scatter = ax.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.title('Scatter Plot of Moon Dataset')
plt.xlabel('First Feature')
plt.ylabel('Second Feature')
colorbar = plt.colorbar(scatter)
colorbar.set_label('Class Label')
plt.show()

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

# Let's write Accuracy score metrix

count_of_correct_pred = 0
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        count_of_correct_pred += 1

accuracy_score = count_of_correct_pred/len(y_pred)

plt.figure(figsize=(6, 9), dpi=100)
plt.bar(['Accuracy'], [accuracy_score], color='green')
plt.ylabel('Accuracy')
plt.title('Accuracy score')
plt.ylim(0, 1)
plt.text(0, accuracy_score, f'{accuracy_score:.2f}', ha='center', va='bottom', fontsize=16, color='black')
plt.show()

# Let's look at the dependce of quality on the number of neighbors

metrics = []
for n in [i for i in range(1, 41)]:
    knn1 = KNeighborsRegressor(n_neighbors = n)
    scores = cross_val_score(knn1, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    metrics.append(np.mean(scores))

plt.figure(figsize=(9, 6), dpi=100)    
plt.plot([i for i in range(1, 41)], metrics)
plt.title('Dependce of quality on the number of neighbors')
plt.ylabel('Negative mean squared error')
plt.xlabel('Number of neightbors')
plt.show()

"""
Let's write the train_grid_search function.
The function accepts a selection as input. It should create a Grid Search object C V,
which will iterate over neighbors from one to 20. Train GridSearchCV.
The function should return the mem_test_score key value for the cv_results_ attribute in the GridSearchCV class.
This field contains information about the metric value for each parameter.
"""

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

