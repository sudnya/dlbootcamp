import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW          = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train   = X.shape[0]
  loss        = 0.0

  for i in xrange(num_train):
    scores              = X[i].dot(W)
    correct_class_score = scores[y[i]]
    
    for j in xrange(num_classes):
      if j == y[i]: #don't do anything if correct
        continue
      
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      if margin > 0:
        loss        += margin
        dW[:, j]    += 1.0*X[i,:]
        dW[:, y[i]] += -1.0*X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs and outputs are the same as svm_loss_naive.
  """
  loss  = 0.0
  dW    = np.zeros(W.shape) # initialize the gradient as zero
  delta = 1.0

  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  num_classes = W.shape[1]
  num_train   = X.shape[0]
  
  scores = X.dot(W)
  #print "scores is ", scores.shape
  correct_scores = scores[np.arange(num_train), y]
  #print "correct scores ", correct_scores.shape

  margin = np.maximum(0, scores - correct_scores[:,np.newaxis] + delta) #TODO Remember this newaxis 
  margin[np.arange(num_train), y] = 0.0 #otherwise every sample has an extra 1 for the correct y

  #print "margin ", margin.shape

  temp = np.zeros((num_train, num_classes))
  temp[margin > 0] = 1

  incorrect_scores = np.sum(margin, axis=1)
  #print "incorrect scores ", incorrect_scores.shape
  temp[np.arange(num_train), y] = -incorrect_scores
  #print "temp ", temp.shape
  #print "X ", X.shape

  loss += np.sum(margin)
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #compensate by deducting the wrong ones
  dW = np.dot(X.T, temp)
  dW   /= num_train
  dW   += reg*W

  return loss, dW
