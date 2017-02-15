import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def reLu(self, y):
      return np.maximum(0.0, y)

  def reLuDerivative(self, x):
      retVal = x.copy()
      retVal[retVal > 0]  = 1.0
      retVal[retVal <= 0] = 0.0
      return retVal


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D   = X.shape
    
    loss      = 0.0

    # Compute the forward pass
    scores = None
    p0 = np.dot(X, W1)
    p1 = p0 + b1
    p2 = self.reLu(p1)
    p3 = np.dot(p2, W2)
    p4 = p3 + b2

    scores = p3 + b2

    #softmax
    #p4       -= np.max(p4, axis=1, keepdims=True)
    stableP4  = np.exp(p4)

    sMaxDeno  = np.sum(stableP4, axis=1, keepdims=True)
    probs     = stableP4/sMaxDeno
    probs_correct = probs[np.arange(N), y]

    # Finish the forward pass, and compute the loss. This should include  
    # both the data loss and L2 regularization for W1 and W2. Store the result  
    # in the variable loss, which should be a scalar. Use the Softmax           
    # classifier loss. So that your results match ours, multiply the            
    # regularization loss by 0.5                                                
    loss      = -np.sum(np.log(probs_correct))
    loss     /=N 
    loss     += 0.5 * reg * np.sum(W1*W1)
    loss     += 0.5 * reg * np.sum(W2*W2)
    #loss     += 0.5 * reg * np.sum(b1*b1) #TODO: why not?
    #loss     += 0.5 * reg * np.sum(b2*b2)

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Backward pass: compute gradients
    grads = {}

    probs[np.arange(N), y] -= 1 #account for the correct ones
    dLdy = probs
    dLdy /= N

    dLdW2   = p2.T.dot(dLdy)
    dLdb2   = np.sum(dLdy, axis=0)
    dLdp2   = dLdy.dot(W2.T)
    #dLdReLu = self.reLuDerivative(dLdp2) #TODO : make this work instead
    #dLdW1   = X.T.dot(dLdReLu)
    #dLdb1   = np.sum(dLdReLu, axis=0)
    
    dLdp2[p2 <= 0] = 0
    dLdW1   = X.T.dot(dLdp2)
    dLdb1   = np.sum(dLdp2, axis=0)

    # Compute the backward pass, computing the derivatives of the weights 
    # and biases. Store the results in the grads dictionary. For example,       
    # grads['W1'] should store the gradient on W1, and be a matrix of same size 
    grads['W2'] = dLdW2 + (reg*W2)
    grads['W1'] = dLdW1 + (reg*W1)

    grads['b1'] = dLdb1
    grads['b2'] = dLdb2

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      temp = np.random.choice(num_train, batch_size, replace=True)
      X_batch = X[temp]
      y_batch = y[temp]
      #print "batch is ", X_batch.shape, y_batch.shape

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W1'] += -learning_rate * grads['W1'] 
      self.params['W2'] += -learning_rate * grads['W2'] 
      self.params['b1'] += -learning_rate * grads['b1'] 
      self.params['b2'] += -learning_rate * grads['b2'] 

      #print "grads W2 " , grads['W1'].shape
      #print "grads W1 " , grads['W2'].shape
      #print "weights ", self.params['W1'].shape , " -- ", self.params['W2'].shape

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = np.mean(self.predict(X_batch) == y_batch)
        val_acc = np.mean(self.predict(X_val) == y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    y_pred = np.argmax(self.loss(X), axis=1)

    return y_pred


