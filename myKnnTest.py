# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print 'Training data shape: ', X_train.shape
print 'Training labels shape: ', y_train.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print X_train.shape, X_test.shape

from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)


# Open cs231n/classifiers/k_nearest_neighbor.py and implement
# compute_distances_two_loops.

# Test your implementation:
dists = classifier.compute_distances_two_loops(X_test)
#print dists.shape
## Now implement the function predict_labels and run the code below:
## We use k = 1 (which is Nearest Neighbor).
#y_test_pred = classifier.predict_labels(dists, k=1)
#
## Compute and print the fraction of correctly predicted examples
#num_correct = np.sum(y_test_pred == y_test)
#accuracy = float(num_correct) / num_test
#print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)
#dists_one = classifier.compute_distances_one_loop(X_test)
#
## To ensure that our vectorized implementation is correct, we make sure that it
## agrees with the naive implementation. There are many ways to decide whether
#difference = np.linalg.norm(dists - dists_one, ord='fro')
#print 'Difference was: %f' % (difference, )
#if difference < 0.001:
#    print 'Good! The distance matrices are the same'
#else:
#    print 'Uh-oh! The distance matrices are different'

dists_two = classifier.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
    print 'Good! The distance matrices are the same'
else:
    print 'Uh-oh! The distance matrices are different'

def crossValidate(X_fold, y_fold, k, idx):
    #print "Use idx ", idx , " for crossvalidation"
    #X_train = np.array(len(X_fold)-1)
    #X_cross = np.array(l)
    #y_train = np.array(len(y_fold)-1)
    #y_cross = np.array(len(y_fold))

    for i in xrange(0, len(X_fold)):
        if i == idx:
            X_cross = X_fold[i]
            y_cross = y_fold[i]
        else:
            X_train = np.vstack(X_fold[0:i]+X_fold[i+1:])
            y_train = np.hstack(y_fold[0:i]+y_fold[i+1:])

#    print "dim train ", X_train.shape
#    print "dim cross ", X_cross.shape
#    print "dim y train ", y_train.shape
#    print "dim y cross ", y_cross.shape
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_no_loops(X_cross)
    y_cross_pred = classifier.predict_labels(dists, k)

    num_correct = np.sum(y_cross_pred == y_cross)
    print "cross val has ", y_cross.shape
    accuracy = float(num_correct) / len(y_cross)
    return accuracy


def kFoldValidation(X_folds, y_folds, k, folds):
    accuracyFold = []
    for i in range(0, folds):
#        print "Withhold idx ", i , " for validation"
        accuracyFold.append(crossValidate(X_folds, y_folds, k, i))

    return accuracyFold

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []

for i in range(0, len(k_choices)):
#    print "Evaluating for k = ", k_choices[i]
    k_to_accuracies[k_choices[i]] = kFoldValidation(X_train_folds, y_train_folds, k_choices[i], num_folds)

for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)
