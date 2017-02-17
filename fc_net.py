import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg    = reg

    w1 = weight_scale * np.random.randn(input_dim, hidden_dim)
    b1 = np.zeros(hidden_dim)
    
    w2 = weight_scale * np.random.randn(hidden_dim, num_classes)
    b2 = np.zeros(num_classes)

    self.params['W1'] = w1
    self.params['b1'] = b1
    self.params['W2'] = w2
    self.params['b2'] = b2

    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N      = X.shape[0]
    temp   = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))

    p0 = np.dot(temp, W1)
    p1 = p0 + b1
    p2 = np.maximum(0, p1)
    p3 = np.dot(p2, W2)
    p4 = p3 + b2

    scores = p3 + b2

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    
    stableP4      = np.exp(p4)
    sMaxDeno      = np.sum(stableP4, axis=1, keepdims=True)
    probs         = stableP4/sMaxDeno
    probs_correct = probs[np.arange(N), y]
    loss          = -np.sum(np.log(probs_correct))
    loss         /= N
    loss         += 0.5 * self.reg * np.sum(W1*W1)
    loss         += 0.5 * self.reg * np.sum(W2*W2)

    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    grads = {}

    probs[np.arange(N), y] -= 1 #account for the correct ones
    dLdy                    = probs
    dLdy                   /= N

    dLdW2          = p2.T.dot(dLdy)
    dLdb2          = np.sum(dLdy, axis=0)
    dLdp2          = dLdy.dot(W2.T)
    dLdp2[p2 <= 0] = 0
    dLdW1          = temp.T.dot(dLdp2)
    dLdb1          = np.sum(dLdp2, axis=0)

    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #

    grads['W2'] = dLdW2 + (self.reg*W2)
    grads['W1'] = dLdW1 + (self.reg*W1)

    grads['b1'] = dLdb1
    grads['b2'] = dLdb2

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    self.all_dims = []
    self.all_dims.append(input_dim)
    self.all_dims += list(hidden_dims)
    self.all_dims.append(num_classes)

    layer_size = input_dim
    N = self.num_layers

    for i, hd in enumerate(hidden_dims):
        w_key = 'W' + str(i+1)
        b_key = 'b' + str(i+1)
        if self.use_batchnorm:
            gamma_key = 'gamma' + str(i+1)
            beta_key  = 'beta' + str(i+1)
            self.params[gamma_key] = np.ones(hd)
            self.params[beta_key]  = np.zeros(hd)

        self.params[w_key]     = weight_scale * np.random.randn(layer_size, hd)
        self.params[b_key]     = np.zeros(hd)
        layer_size = hd
    self.params['W%d' % N] = weight_scale * np.random.randn(layer_size, num_classes)
    self.params['b%d' % N] = np.zeros(num_classes) 

    #for k, v in self.params.iteritems():
    #    print k
    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    caches = []
    dp_cache = {}
    N = self.num_layers
    loss, grads = 0.0, {}
    prev_activations = X
    for i in range(1, N):
        W_idx        = "W" + str(i)
        b_idx        = "b" + str(i)

        current_W    = self.params[W_idx]
        current_b    = self.params[b_idx]

        if self.use_batchnorm:
            gamma_key = 'gamma' + str(i)
            beta_key  = 'beta' + str(i)
            #print 'Gamma key: ', gamma_key
            #print 'Beta key: ', beta_key
            #print 'X shape: ', prev_activations.shape
            #print 'Current W key: ', W_idx, ' , W shape: ', current_W.shape
            #print 'Current b key: ', b_idx, ' , b shape: ', current_b.shape
            #print 'Current gamma key: ', gamma_key, ' , gamma shape: ', self.params[gamma_key].shape
            #print 'Current beta key: ', beta_key, ' , beta shape: ', self.params[beta_key].shape

            out, cache = affine_batchnorm_relu_forward(prev_activations, current_W, current_b, self.params[gamma_key], self.params[beta_key], self.bn_params[i-1])
        else:
            out, cache  = affine_relu_forward(prev_activations, current_W , current_b)
        
        if self.use_dropout:
            out, dp_cache[i] = dropout_forward(out, self.dropout_param)
        caches.append(cache)
        #dout, cache  = relu_forward(dout)
        prev_activations = out
    out, cache = affine_forward(prev_activations, self.params["W%d"%(N)], self.params["b%d"%(N)])

    scores = out

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    # If test mode return early
    if mode == 'test':
      return scores

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # loss
    caches.reverse()
    loss, dout = softmax_loss(scores, y)

    dx,dw,db = affine_backward(dout, cache)
    grads["W%d"%(N)] = dw + self.reg*self.params['W%d'%(N)]
    grads["b%d"%(N)] = db

    loss         += 0.5 * self.reg * np.sum(np.square(self.params['W%d'%(N)]))

    for i in range(self.num_layers-1, 0, -1):
        W_idx        = "W" + str(i)
        b_idx        = "b" + str(i)

        if self.use_dropout:
            dx = dropout_backward(dx, dp_cache[i])
        
        if self.use_batchnorm:
            gamma_key = 'gamma' + str(i)
            beta_key  = 'beta' + str(i)
            dx,dw,db,dgamma,dbeta = affine_batchnorm_relu_backward(dx, caches.pop(0))
            grads[gamma_key] = dgamma
            grads[beta_key]  = dbeta
            #print "updating gamma and beta for layer H ", i

        else:
            dx,dw,db = affine_relu_backward(dx, caches.pop(0))
        grads[W_idx] = dw + self.reg*self.params[W_idx]
        grads[b_idx] = db
    
        loss         += 0.5 * self.reg * np.sum(np.square(self.params['W%d'%(i)]))
    
    return loss, grads
