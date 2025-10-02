#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from svm_source import *
from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from time import time

scaler = StandardScaler()

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')

import random
import pandas as pd

# Pour utiliser utiliser tous les cœurs disponibles
import os
import joblib


#n_cores = joblib.cpu_count()
#print(f"Nombre de cœurs disponibles : {n_cores}")


#%%
###############################################################################
#               Toy dataset : 2 gaussians
###############################################################################

random.seed(5723)

n1 = 200
n2 = 200
mu1 = [1., 1.]
mu2 = [-1./2, -1./2]
sigma1 = [0.9, 0.9]
sigma2 = [0.9, 0.9]
X1, y1 = rand_bi_gauss(n1, n2, mu1, mu2, sigma1, sigma2)

plt.show()
plt.close("all")
plt.ion()
plt.figure(1, figsize=(15, 5))
plt.title('First data set')
plot_2d(X1, y1)

X_train = X1[::2]
Y_train = y1[::2].astype(int)
X_test = X1[1::2]
Y_test = y1[1::2].astype(int)

# fit the model with linear kernel
clf = SVC(kernel='linear')
clf.fit(X_train, Y_train)

# predict labels for the test data base
y_pred = clf.predict(X_test)

# check your score
score = clf.score(X_test, Y_test)
print('Score : %s' % score)

# display the frontiere
def f(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf.predict(xx.reshape(1, -1))

plt.figure()
frontiere(f, X_train, Y_train, w=None, step=50, alpha_choice=1)

# Same procedure but with a grid search
parameters = {'kernel': ['linear'], 'C': list(np.linspace(0.001, 3, 21))}
clf2 = SVC()
clf_grid = GridSearchCV(clf2, parameters, n_jobs=-1)
clf_grid.fit(X_train, Y_train)

# check your score
print(clf_grid.best_params_)
print('Score : %s' % clf_grid.score(X_test, Y_test))

def f_grid(xx):
    """Classifier: needed to avoid warning due to shape issues"""
    return clf_grid.predict(xx.reshape(1, -1))

# display the frontiere
plt.figure()
frontiere(f_grid, X_train, Y_train, w=None, step=50, alpha_choice=1)

#%%
###############################################################################
#               Iris Dataset
###############################################################################
random.seed(5723)


iris = datasets.load_iris()
X = iris.data
X = scaler.fit_transform(X)
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

# split train test (say 25% for the test)
# You can shuffle and then separate or you can just use train_test_split 
# Whithout shuffling (in that case fix the random state (say to 42) for reproductibility)

# Split train/test (25% pour le test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# Visualization of dataset
cols = [iris.feature_names[0], iris.feature_names[1]]  # noms des 2 premières features
X_train_df = pd.DataFrame(X_train, columns=cols)
X_test_df  = pd.DataFrame(X_test,  columns=cols)

print("\nAperçu X_train:")
print(X_train_df.head())

print("\nAperçu X_test:")
print(X_test_df.head())

###############################################################################
# fit the model with linear vs polynomial kernel
###############################################################################

#%%
# Q1 linear kernel
random.seed(5723)

# Define the grid of C values (200 values between 10^-3 and 10^3 on a log scale)
parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 200))}

# Cross-validation to select the best C
clf_linear = GridSearchCV(svm.SVC(), parameters, n_jobs=-1)
clf_linear.fit(X_train, y_train)

# Print the generalization score on both training and test sets
print('Generalization score for linear kernel: %s, %s' %
      (clf_linear.score(X_train, y_train),
       clf_linear.score(X_test, y_test)))
#%%
# Q2 polynomial kernel
random.seed(5723)

# Define grids for hyperparameters
Cs = list(np.logspace(-3, 3, 5))        # Regularization parameter C
gammas = 10. ** np.arange(1, 2)         # Gamma values (here only 10)
degrees = np.r_[2, 3]                # Polynomial degrees to test

# Grid search for the best hyperparameters with polynomial kernel
parameters = {'kernel': ['poly'], 'C': Cs, 'gamma': gammas, 'degree': degrees}
clf_poly = GridSearchCV(svm.SVC(), parameters, n_jobs=-1)
clf_poly.fit(X_train, y_train)

# Print the best parameters (typo in original: should be clf_poly.best_params_)
print(clf_grid.best_params_)
# Print generalization scores (train/test) for the best polynomial model
print('Generalization score for polynomial kernel: %s, %s' %
      (clf_poly.score(X_train, y_train),
       clf_poly.score(X_test, y_test)))


#%%
random.seed(5723)

# display your results using frontiere (svm_source.py)

def f_linear(xx):
    return clf_linear.predict(xx.reshape(1, -1))

def f_poly(xx):
    return clf_poly.predict(xx.reshape(1, -1))

plt.ion()
plt.figure(figsize=(15, 5))
plt.subplot(131)
plot_2d(X, y)
plt.title("iris dataset")

plt.subplot(132)
frontiere(f_linear, X, y)
plt.title("linear kernel")

plt.subplot(133)
frontiere(f_poly, X, y)

plt.title("polynomial kernel")
plt.tight_layout()
plt.draw()

#%%
###############################################################################
#               SVM GUI
###############################################################################

# please open a terminal and run python svm_gui.py
# Then, play with the applet : generate various datasets and observe the
# different classifiers you can obtain by varying the kernel

#%%
###############################################################################
#               Face Recognition Task
###############################################################################
"""
The dataset used in this example is a preprocessed excerpt
of the "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

  _LFW: http://vis-www.cs.umass.edu/lfw/
"""

####################################################################
# Download the data and unzip; then load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4,
                              color=True, funneled=False, slice_=None,
                              download_if_missing=True)
# data_home='.'

# introspect the images arrays to find the shapes (for plotting)
images = lfw_people.images
n_samples, h, w, n_colors = images.shape

# the label to predict is the id of the person
target_names = lfw_people.target_names.tolist()

####################################################################
random.seed(5723)

# Pick a pair to classify such as
names = ['Tony Blair', 'Colin Powell']
# names = ['Donald Rumsfeld', 'Colin Powell']

idx0 = (lfw_people.target == target_names.index(names[0]))
idx1 = (lfw_people.target == target_names.index(names[1]))
images = np.r_[images[idx0], images[idx1]]
n_samples = images.shape[0]
y = np.r_[np.zeros(np.sum(idx0)), np.ones(np.sum(idx1))].astype(np.int64)

# plot a sample set of the data
plot_gallery(images, np.arange(12))
plt.show()

#%%
####################################################################
# Extract features
random.seed(5723)
# features using only illuminations
X = (np.mean(images, axis=3)).reshape(n_samples, -1)

# # or compute features using colors (3 times more features)
# X = images.copy().reshape(n_samples, -1)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

#%%
random.seed(5723)

####################################################################
# Split data into a half training and half test set
# X_train, X_test, y_train, y_test, images_train, images_test = \
#    train_test_split(X, y, images, test_size=0.5, random_state=0)
# X_train, X_test, y_train, y_test = \
#    train_test_split(X, y, test_size=0.5, random_state=0)

indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:X.shape[0] // 2], indices[X.shape[0] // 2:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]
images_train, images_test = images[
    train_idx, :, :, :], images[test_idx, :, :, :]

####################################################################
# Quantitative evaluation of the model quality on the test set

#%%
random.seed(5723)

# Q4
print("--- Linear kernel ---")
print("Fitting the classifier to the training set")
t0 = time()

# fit a classifier (linear) and test all the Cs
Cs = 10. ** np.arange(-5, 6)
train_scores = []
test_scores = []

for C in Cs:
    clf_tmp = svm.SVC(kernel='linear', C=C)
    clf_tmp.fit(X_train, y_train)
    train_scores.append(clf_tmp.score(X_train, y_train))
    test_scores.append(clf_tmp.score(X_test, y_test))

ind_test = np.argmax(test_scores)
print("Best C (test): {}".format(Cs[ind_test]))
print("Best train score: {:.4f}".format(np.max(train_scores)))
print("Best test score: {:.4f}".format(np.max(test_scores)))
print("Overfitting gap at best C: {:.4f}".format(train_scores[ind_test] - test_scores[ind_test]))
print("Done in %0.3fs" % (time() - t0))

plt.figure(figsize=(10,6))
plt.plot(Cs, train_scores, marker='o', label="Train score")
plt.plot(Cs, test_scores, marker='s', label="Test score")
plt.plot(Cs, 1 - np.array(test_scores), marker='^', linestyle="--", label="Test error")

plt.xscale("log")
plt.xlabel("Paramètre de régularisation C")
plt.ylabel("Score / Erreur")
plt.title("Influence du paramètre C (SVM linéaire)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# predict labels for the X_test images with the best classifier
print("Predicting the people names on the testing set")
clf = svm.SVC(kernel='linear', C=Cs[ind_test])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("done in %0.3fs" % (time() - t0))
# The chance level is the accuracy that will be reached when constantly predicting the majority class.
print("Chance level : %s" % max(np.mean(y), 1. - np.mean(y)))
print("Accuracy : %s" % clf.score(X_test, y_test))
