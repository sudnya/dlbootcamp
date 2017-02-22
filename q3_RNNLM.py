import getpass
import sys
import time

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample

import tensorflow as tf
#from tensorflow.python.ops.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import sequence_loss
from model import LanguageModel

from q2_initialization import xavier_weight_init

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  embed_size = 50
  hidden_size = 100
  num_steps = 10
  max_epochs = 16
  early_stopping = 2
  dropout = 0.9
  lr = 0.001

class RNNLM_Model(LanguageModel):

  def load_data(self, debug=False):
    """Loads starter word-vectors and train/dev/test data."""
    self.vocab = Vocab()
    self.vocab.construct(get_ptb_dataset('train'))
    self.encoded_train = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('train')],
        dtype=np.int32)
    self.encoded_valid = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
        dtype=np.int32)
    self.encoded_test = np.array(
        [self.vocab.encode(word) for word in get_ptb_dataset('test')],
        dtype=np.int32)
    #debug = True
    if debug:
      num_debug = 128#1024
      self.encoded_train = self.encoded_train[:num_debug]
      self.encoded_valid = self.encoded_valid[:num_debug]
      self.encoded_test = self.encoded_test[:num_debug]
      print "vocab size ", len(self.vocab)
      print "training samples ", len(self.encoded_train)

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph.
    (When None is in a placeholder's shape, it's flexible)

    input_placeholder: Input placeholder tensor of shape
                       (None, num_steps), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, num_steps), type #TODO: wrong tf.float32 --> should be tf.int32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    self.input_placeholder   = tf.placeholder(name="inputs",  dtype=tf.int32, shape=(None, self.config.num_steps))
    self.labels_placeholder  = tf.placeholder(name="outputs", dtype=tf.int32, shape=(None, self.config.num_steps))
    self.dropout_placeholder = tf.placeholder(name="dropout", dtype=tf.float32)
    ### END YOUR CODE
  
  def add_embedding(self):
    """Add embedding layer.

    Hint: This layer should use the input_placeholder to index into the
          embedding.
    Hint: You might find tf.nn.embedding_lookup useful.
    Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
    Hint: Check the last slide from the TensorFlow lecture.
    ---->
    From the slides
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_input)
    <-----
    Hint: Here are the dimensions of the variables you will need to create:

      L: (len(self.vocab), embed_size)

    Returns:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    """
    # The embedding lookup is currently only implemented for the CPU
    with tf.device('/cpu:0'):
      ### YOUR CODE HERE
      
      #current mini batch
      #for t in range(self.config.num_steps):
          # t = 0 to n time steps (processed at once)
          # no need! understand tf.nn.embedding_lookup and you will know why!
      with tf.variable_scope('embed') as embed_scope:
          embeddings      = tf.get_variable("embeddings", shape=[len(self.vocab), self.config.embed_size], initializer=xavier_weight_init())
          embedded_inputs = tf.nn.embedding_lookup(params=embeddings, ids=tf.transpose(self.input_placeholder)) #this miniB
          print "embedded look up -> ", embedded_inputs.get_shape()
          # dims are embedded look up ->  (10, ?, 50)
          embedded_inputs = tf.split(embedded_inputs, self.config.num_steps, axis=0) # each 'cell' in the RNN
          print "split gives ", len(embedded_inputs) , " entries of dim ", embedded_inputs[0].shape
          #inputs = embedded_inputs
          #print "split embedded look up -> ", len(embedded_inputs) , "  ", embedded_inputs[0].shape
          inputs          = map(lambda x: tf.squeeze (x, axis=[0]), embedded_inputs) #get rid of the ? dim above
          #print "inputs after tf.squeeze are: ", inputs

      #raise NotImplementedError
      ### END YOUR CODE
      print "at the end of embedLayer ", len(inputs), "   ", inputs[0].shape
      return inputs

  def add_projection(self, rnn_outputs):
    """Adds a projection layer.

    The projection layer transforms the hidden representation to a distribution
    over the vocabulary.

    Hint: Here are the dimensions of the variables you will need to
          create 
          
          U:   (hidden_size, len(vocab))
          b_2: (len(vocab),)

    Args:
      rnn_outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, embed_size).
                   wrong!!!! hidden_size not embed_size
    Returns:
      outputs: List of length num_steps, each a tensor of shape
               (batch_size, len(vocab)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    #this is W_hy
    print "rnn_outputs dims ", rnn_outputs[0].shape
    U   = tf.get_variable("U", shape=[self.config.hidden_size, len(self.vocab)], initializer=xavier_weight_init())
    b_2 = tf.get_variable("b_2", shape=[len(self.vocab)], initializer=xavier_weight_init())
    outputs = []
    for t_step in range(len(rnn_outputs)):
        outputs.append(tf.matmul(tf.nn.dropout(rnn_outputs[t_step], self.dropout_placeholder), U) + b_2)

    ### END YOUR CODE
    return outputs

  def add_loss_op(self, output):
    """Adds loss ops to the computational graph.

    Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss. 

    Args:
      output: A tensor of shape (None, self.vocab)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    b_size  = self.config.batch_size
    n_steps = self.config.num_steps
    targets = [tf.reshape(self.labels_placeholder, [-1])]
    weights = [tf.ones([b_size*n_steps])]
    print "\n\nLoss Op: "
    print "logits ", len(output), " - ", output[0].shape
    t = tf.reshape(self.labels_placeholder, [b_size, n_steps])
    print "labels ", t
    #print "weights  ",  
    w = tf.ones([b_size, n_steps])
    print "weights ", w
    f = tf.reshape(output, [b_size, n_steps, -1])
    print "reshaped ", f
    s2s_loss = sequence_loss(logits=f, targets=t, weights=w)
    tf.add_to_collection('total_loss', s2s_loss)
    loss = s2s_loss
    print loss
    #raise NotImplementedError
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train. See 

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Hint: Use tf.train.AdamOptimizer for this model.
          Calling optimizer.minimize() will return a train_op object.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
    ### YOUR CODE HERE
    optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
    train_op  = optimizer.minimize(loss)
    #raise NotImplementedError
    ### END YOUR CODE
    return train_op
  
  def __init__(self, config):
    self.config = config
    self.load_data(debug=False)
    self.add_placeholders()
    self.inputs = self.add_embedding()
    self.rnn_outputs = self.add_model(self.inputs)
    self.outputs = self.add_projection(self.rnn_outputs)
    print 'outputs shape: ', self.outputs[0].shape
    print len(self.outputs)
  
    # We want to check how well we correctly predict the next word
    # We cast o to float64 as there are numerical issues at hand
    # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
    self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
    print "0th pred ", self.predictions[0]
    #print "vocab ", self.vocab
    # Reshape the output into len(vocab) sized chunks - the -1 says as many as
    # needed to evenly divide
    #outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, len(self.vocab)])
    #self.calculate_loss = self.add_loss_op(self.outputs)

    #TODO: API changed for seq to seq loss!
    #outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, len(self.vocab)])
    self.calculate_loss = self.add_loss_op(self.outputs)
    self.train_step = self.add_training_op(self.calculate_loss)


  def add_model(self, inputs):
    """Creates the RNN LM model.

    In the space provided below, you need to implement the equations for the
    RNN LM model. Note that you may NOT use built in rnn_cell functions from
    tensorflow.

    Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
          initial state for the RNN. Add this to self as instance variable

          self.initial_state
  
          (Don't change variable name)
    Hint: Add the last RNN output to self as instance variable

          self.final_state

          (Don't change variable name)
    Hint: Make sure to apply dropout to the inputs and the outputs.
    Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
    Hint: Perform an explicit for-loop over inputs. You can use
          scope.reuse_variables() to ensure that the weights used at each
          iteration (each time-step) are the same. (Make sure you don't call
          this for iteration 0 though or nothing will be initialized!)
    Hint: Here are the dimensions of the various variables you will need to
          create:
      
          H: (hidden_size, hidden_size) 
          I: (embed_size, hidden_size)
          b_1: (hidden_size,)

    Args:
      inputs: List of length num_steps, each of whose elements should be
              a tensor of shape (batch_size, embed_size).
    Returns:
      outputs: List of length num_steps, each of whose elements should be
               a tensor of shape (batch_size, hidden_size)
    """
    ### YOUR CODE HERE
    #raise NotImplementedError
    num_steps = len(inputs)
    b_size, e_size = self.config.batch_size, self.config.embed_size
    print num_steps , " steps "
    print b_size, " seq "
    print e_size, "embeddings"


    h_size = self.config.hidden_size

    self.initial_state = tf.zeros(shape=(b_size, h_size))
    embeddings = tf.get_collection('embeddings', 'embed_scope')
    rnn_outputs = []

    with tf.variable_scope('RNN') as scope:
        #scope.reuse_variables()
        H   = tf.get_variable(name='H',   dtype=tf.float32, shape=[h_size, h_size])
        I   = tf.get_variable(name='I',dtype=tf.float32, shape=[e_size, h_size])
        b_1 = tf.get_variable(name='b_1', dtype=tf.float32, shape=[h_size])


        #at t = 0
        rnn_outputs.append(tf.sigmoid( tf.matmul(self.initial_state, H) + tf.matmul(inputs[0], I) + b_1) )
        #remaining
        for i in range(1, num_steps):
            rnn_outputs.append(tf.sigmoid( tf.matmul(rnn_outputs[i-1], H) + tf.matmul(inputs[i], I) + b_1 ))

    
        self.final_state = rnn_outputs[num_steps-1]

    ### END YOUR CODE
    return rnn_outputs


  def run_epoch(self, session, data, train_op=None, verbose=10):
    config = self.config
    dp = config.dropout
    if not train_op:
      train_op = tf.no_op()
      dp = 1
    total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
    total_loss = []
    state = self.initial_state.eval()
    for step, (x, y) in enumerate(
      ptb_iterator(data, config.batch_size, config.num_steps)):
      # We need to pass in the initial state and retrieve the final state to give
      # the RNN proper history
      feed = {self.input_placeholder: x,
              self.labels_placeholder: y,
              self.initial_state: state,
              self.dropout_placeholder: dp}
      loss, state, _ = session.run(
          [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
      total_loss.append(loss)
      if verbose and step % verbose == 0:
          sys.stdout.write('\r{} / {} : pp = {}'.format(
              step, total_steps, np.exp(np.mean(total_loss))))
          sys.stdout.flush()
    if verbose:
      sys.stdout.write('\r')
    return np.exp(np.mean(total_loss))

def generate_text(session, model, config, starting_text='<eos>',
                  stop_length=100, stop_tokens=None, temp=1.0):
  """Generate text from the model.

  Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
        that you will need to use model.initial_state as a key to feed_dict
  Hint: Fetch model.final_state and model.predictions[-1]. (You set
        model.final_state in add_model() and model.predictions is set in
        __init__)
  Hint: Store the outputs of running the model in local variables state and
        y_pred (used in the pre-implemented parts of this function.)

  Args:
    session: tf.Session() object
    model: Object of type RNNLM_Model
    config: A Config() object
    starting_text: Initial text passed to model.
  Returns:
    output: List of word idxs
  """
  state = model.initial_state.eval()
  # Imagine tokens as a batch size of one, length of len(tokens[0])
  tokens = [model.vocab.encode(word) for word in starting_text.split()]
  for i in xrange(stop_length):
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append(next_word_idx)
    if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
      break
  output = [model.vocab.decode(word_idx) for word_idx in tokens]
  return output

def generate_sentence(session, model, config, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)

def test_RNNLM():
  config = Config()
  gen_config = deepcopy(config)
  gen_config.batch_size = gen_config.num_steps = 1
# We create the training model and generative model
  with tf.variable_scope('RNNLM', reuse=None) as scope:
   model = RNNLM_Model(config)
  # This instructs gen_model to reuse the same variables as the model above
  with tf.variable_scope('RNNLM', reuse=True) as scope:
   gen_model = RNNLM_Model(gen_config)

  # We create the training model and generative model
#  with tf.variable_scope('RNNLM') as scope:
#    model = RNNLM_Model(config)
#    # This instructs gen_model to reuse the same variables as the model above
#    scope.reuse_variables()
#    gen_model = RNNLM_Model(gen_config)

  init = tf.initialize_all_variables()
  saver = tf.train.Saver()

  with tf.Session() as session:
    best_val_pp = float('inf')
    best_val_epoch = 0
  
    session.run(init)
    for epoch in xrange(config.max_epochs):
      print 'Epoch {}'.format(epoch)
      start = time.time()
      ###
      train_pp = model.run_epoch(
          session, model.encoded_train,
          train_op=model.train_step)
      valid_pp = model.run_epoch(session, model.encoded_valid)
      print 'Training perplexity: {}'.format(train_pp)
      print 'Validation perplexity: {}'.format(valid_pp)
      if valid_pp < best_val_pp:
        best_val_pp = valid_pp
        best_val_epoch = epoch
        saver.save(session, './ptb_rnnlm.weights')
      if epoch - best_val_epoch > config.early_stopping:
        break
      print 'Total time: {}'.format(time.time() - start)
      
    saver.restore(session, 'ptb_rnnlm.weights')
    test_pp = model.run_epoch(session, model.encoded_test)
    print '=-=' * 5
    print 'Test perplexity: {}'.format(test_pp)
    print '=-=' * 5
    starting_text = 'in palo alto'
    while starting_text:
      print ' '.join(generate_sentence(
          session, gen_model, gen_config, starting_text=starting_text, temp=1.0))
      starting_text = raw_input('> ')

if __name__ == "__main__":
    test_RNNLM()
