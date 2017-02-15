import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  num_train    = X.shape[0]
  num_classes  = W.shape[1]
  num_features = W.shape[0]

  dW = np.zeros_like(W.shape)
  dLdW = np.zeros(W.shape)
  dLdy = np.zeros((num_features, num_classes))

  for i in range(0, num_train):
      #per sample
      stableP1  = X[i].dot(W)
      stableP1 -= np.max(stableP1)
      p2           = np.exp(stableP1)
      allExp       = np.sum(p2)
      p3 = p2/allExp
      p3_correct = p3[y[i]]
      #print "X ", X.shape
      #print "dLdW ", dLdW.shape

      dLdW[: , y[i]] += -X[i]
      loss += -np.log(p3_correct)

      for j in range(0, num_classes):
          
          dLdW[:, j] += p3[j]*X[i]
      
      
  loss /= num_train
  loss += 0.5 * reg* np.sum(W*W)

  dW = dLdW/num_train + reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss        = 0.0
  num_train   = X.shape[0]
  num_classes = W.shape[1]

  dW          = np.zeros(W.shape)
  stableP1    = X.dot(W)
  stableP1   -= np.max(stableP1, axis=1, keepdims=True) #TODO: don't blow up
  p2          = np.exp(stableP1)

  allExp      = np.sum(p2, axis=1, keepdims=True)
  p3 = p2/allExp

  p3_correct = p3[np.arange(p3.shape[0]), y]

  loss        = -np.sum(np.log(p3_correct))
  loss       /= num_train
  loss       += 0.5 * reg*np.sum(W*W)

  p3[np.arange(p3.shape[0]),y] -= 1
  
  #print "dldy ", dLdy.shape
  #print "X ", X.shape
  #print "W ", W.shape

  dLdW = (X.T).dot(p3) 
  #print "dLdW ", dLdW.shape
  dW = dLdW/num_train + reg*W
  

  return loss, dW

