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
    param_grid = {'n_neighbors': list(range(1, 21))}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X, y)
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    return mean_test_scores

mean_test_scores = []
for i in range(1, 21):
    mean_test_score = train_grid_search(X_train, y_train)
    mean_test_scores.append(mean_test_score)

mean_test_scores = np.array(mean_test_scores)
mean_scores_mean = np.mean(mean_test_scores, axis=0)

plt.figure(figsize=(9,6), dpi=100)
plt.plot(np.arange(1, 21), mean_scores_mean)
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean Test Score')
plt.title('Mean Test Score vs Number of Neighbors')
plt.grid(True)
plt.show()

"""
This is second part of our case
We will work with 'MNIST data'
We need calculate accuracy score
"""

# load python 'mnist' if you have not it

!pip install python-mnist
!mkdir dir_with_mnist_data_files

import requests

image_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
label_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

response = requests.get(image_url)
with open("dir_with_mnist_data_files/train-images-idx3-ubyte.gz", "wb") as file:
    file.write(response.content)

response = requests.get(label_url)
with open("dir_with_mnist_data_files/train-labels-idx1-ubyte.gz", "wb") as file:
    file.write(response.content)


from mnist import MNIST

mndata = MNIST('./dir_with_mnist_data_files', gz=True)
images, labels = mndata.load_training()
images, labels = np.array(images), np.array(labels)

plt.figure(figsize=(8,8), dpi=100)
plt.imshow(images[0].reshape(28, 28))

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=10)

knn_mnist = KNeighborsClassifier(n_neighbors=30)
knn_mnist.fit(X_train, y_train)
y_pred_mnist = knn_mnist.predict(X_test)
accuracy_score_mnist = accuracy(y_test, y_pred_mnist)
accuracy_score_mnist

plt.figure(figsize=(6, 9), dpi=100)
plt.bar(['Accuracy'], [accuracy_score_mnist], color=(0.2, 0.4, 0.6))
plt.ylabel('Accuracy')
plt.title('Accuracy score')
plt.ylim(0, 1)
plt.text(0, accuracy_score, f'{accuracy_score_mnist:.2f}', ha='center', fontsize=16, color='black')
plt.show()
