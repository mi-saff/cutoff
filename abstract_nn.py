import os
import getpass
import sys
import time

import numpy as np
import tensorflow as tf
from utils import data_iterator

from model import Model
import clean_data


def get_data():
    input_data = clean_data.cleaned_data()
    inputs = []
    labels = []
    for key in input_data:
      inputs.append(input_data[key][0])
      labels.append(input_data[key][1])
    return inputs, labels

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 50
  label_size = 20
  hidden_size = 10
  max_epochs = 24
  early_stopping = 2
  dropout = 0.0
  lr = 0.001
  l2 = 0
  feature_size = 40
  window_size = 1

class SimpleModel(Model):
  """Implements a NER (Named Entity Recognition) model.

  This class implements a deep network for named entity recognition. It
  inherits from LanguageModel, which has an add_embedding method in addition to
  the standard Model method.
  """

  def load_data(self, debug=False):
    self.X_train, self.y_train = get_data()
    if debug:
      self.X_train = self.X_train[:1024]
      self.y_train = self.y_train[:1024]
    with open("train_example.txt", "w") as fp:
        fp.write(str(self.X_train))
        fp.write(str(self.y_train))
    # Load the dev set (for tuning hyperparameters)
    self.X_dev, self.y_dev = get_data()
    if debug:
      self.X_dev = self.X_dev[:1024]
      self.y_dev = self.y_dev[:1024]

    # Load the test set (dummy labels only)
    self.X_test, self.y_test = get_data()

  def add_placeholders(self):
    """Generate placeholder variables to represent the input tensors

    These placeholders are used as inputs by the rest of the model building
    code and will be fed data during training.  Note that when "None" is in a
    placeholder's shape, it's flexible

    Adds following nodes to the computational graph

    input_placeholder: Input placeholder tensor of shape
                       (None, window_size), type tf.int32
    labels_placeholder: Labels placeholder tensor of shape
                        (None, label_size), type tf.float32
    dropout_placeholder: Dropout value placeholder (scalar),
                         type tf.float32

    Add these placeholders to self as the instance variables
  
      self.input_placeholder
      self.labels_placeholder
      self.dropout_placeholder

    (Don't change the variable names)
    """
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(tf.float32, shape=[None, Config.feature_size], name='input_placeholder')
    self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, Config.label_size], name='labels_placeholder')
    self.dropout_placeholder = tf.placeholder(tf.float32, name='dropout_placeholder') 

    ### END YOUR CODE

  def create_feed_dict(self, input_batch, dropout, label_batch=None):
    """Creates the feed_dict for softmax classifier.

    A feed_dict takes the form of:

    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }


    Hint: The keys for the feed_dict should be a subset of the placeholder
          tensors created in add_placeholders.
    Hint: When label_batch is None, don't add a labels entry to the feed_dict.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
      dropout: The dropout value.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    ### YOUR CODE HERE
    feed_dict = {
        self.input_placeholder: input_batch,
        self.dropout_placeholder: dropout
    }
    if label_batch is not None:
      feed_dict[self.labels_placeholder] = label_batch
    ### END YOUR CODE
    return feed_dict

  def add_model(self):
    """Adds the 1-hidden-layer NN.

    Hint: Use a variable_scope (e.g. "Layer") for the first hidden layer, and
          another variable_scope (e.g. "Softmax") for the linear transformation
          preceding the softmax. Make sure to use the xavier_weight_init you
          defined in the previous part to initialize weights.
    Hint: Make sure to add in regularization and dropout to this network.
          Regularization should be an addition to the cost function, defined in add_loss_op,
          while dropout should be added after both variable scopes.
    Hint: You might consider using a tensorflow Graph Collection (e.g
          "total_loss") to collect the regularization and loss terms (which you
          will add in add_loss_op below).
    Hint: Here are the dimensions of the various variables you will need to
          create

          W:  (window_size*embed_size, hidden_size)
          b1: (hidden_size,)
          U:  (hidden_size, label_size)
          b2: (label_size)

    https://www.tensorflow.org/programmers_guide/graphs#what_is_a_wzxhzdk22wzxhzdk23tfgraphwzxhzdk24wzxhzdk25
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
    ### YOUR CODE HERE
    loss_function = lambda x : (Config.l2 / 2) * (tf.reduce_sum(x ** 2) / 2)

    with tf.variable_scope("Layer") as scope:
      W = tf.Variable(tf.truncated_normal([self.config.feature_size, self.config.hidden_size]), name="W", dtype=tf.float32)
      b1 = tf.Variable(tf.zeros([self.config.hidden_size], name="b1", dtype=tf.float32))
      h = tf.tanh(tf.add(tf.matmul(self.input_placeholder,W), b1))
      dropout_h = tf.nn.dropout(h, self.dropout_placeholder)

    with tf.variable_scope("Softmax") as scope:
      U = tf.Variable(tf.truncated_normal([self.config.hidden_size, self.config.label_size]), name="U", dtype=tf.float32)
      b2 = tf.Variable(tf.zeros([self.config.label_size], name="b2", dtype=tf.float32))

    output = tf.add(tf.matmul(dropout_h, U), b2)
    output_dropout = tf.nn.dropout(output, self.dropout_placeholder)

    #add regularization loss
    tf.add_to_collection("total_loss", loss_function(U))


    ### END YOUR CODE
    return output_dropout

  def add_loss_op(self, pred):
    """Adds cross_entropy_loss ops to the computational graph.

    Hint: You can use tf.nn.softmax_cross_entropy_with_logits to simplify your
          implementation. You might find tf.reduce_mean useful.
    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
    ### YOUR CODE HERE
    stabilize = lambda x: x - tf.reduce_max(x, reduction_indices=[1], keep_dims=True)
    j_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=stabilize(pred), labels=self.labels_placeholder))
    loss = tf.add(j_loss, tf.add_n(tf.get_collection("total_loss")))
    ### END YOUR CODE
    return loss

  def add_training_op(self, loss):
    with tf.name_scope("train") as scope:
      opt_type = tf.train.AdamOptimizer(self.config.lr)
      train_op = opt_type.minimize(loss)
    return train_op

  def __init__(self, config):
    """Constructs the network using the helper functions defined above."""
    self.config = config
    self.load_data()
    self.add_placeholders()
    y = self.add_model()

    self.loss = self.add_loss_op(y)
    self.predictions = tf.nn.softmax(y)
    one_hot_prediction = tf.argmax(self.predictions, 1)
    correct_prediction = tf.equal(
        tf.argmax(self.labels_placeholder, 1), one_hot_prediction)
    self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))
    self.train_op = self.add_training_op(self.loss)

  def run_epoch(self, session, input_data, input_labels,
                shuffle=True, verbose=True):
    orig_X, orig_y = input_data, input_labels
    dp = self.config.dropout
    # We're interested in keeping track of the loss and accuracy during training
    total_loss = []
    total_correct_examples = 0
    total_processed_examples = 0
    total_steps = len(orig_X) / self.config.batch_size
    for step, (x, y) in enumerate(
      data_iterator(orig_X, orig_y, batch_size=self.config.batch_size,
                   label_size=self.config.label_size)):
      feed = self.create_feed_dict(input_batch=x, dropout=dp, label_batch=y)
      loss, total_correct, _ = session.run(
          [self.loss, self.correct_predictions, self.train_op],
          feed_dict=feed)
      total_processed_examples += len(x)
      total_correct_examples += total_correct
      total_loss.append(loss)
      ##
      if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : loss = {}'.format(
            step, total_steps, np.mean(total_loss)))
        sys.stdout.flush()
    if verbose:
        sys.stdout.write('\r')
        sys.stdout.flush()
    return np.mean(total_loss), total_correct_examples / float(total_processed_examples)

  def predict(self, session, X, y=None):
    """Make predictions from the provided model."""
    # If y is given, the loss is also calculated
    # We deactivate dropout by setting it to 1
    dp = 1
    losses = []
    results = []
    if np.any(y):
        data = data_iterator(X, y, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    else:
        data = data_iterator(X, batch_size=self.config.batch_size,
                             label_size=self.config.label_size, shuffle=False)
    for step, (x, y) in enumerate(data):
      feed = self.create_feed_dict(input_batch=x, dropout=dp)
      if np.any(y):
        feed[self.labels_placeholder] = y
        loss, preds = session.run(
            [self.loss, self.predictions], feed_dict=feed)
        losses.append(loss)
      else:
        preds = session.run(self.predictions, feed_dict=feed)
      predicted_indices = preds.argmax(axis=1)
      results.extend(predicted_indices)
    return np.mean(losses), results

def save_predictions(predictions, filename):
  """Saves predictions to provided file."""
  with open(filename, "wb") as f:
    for prediction in predictions:
      f.write(str(prediction) + "\n")

def test_model():
  """Test NER model implementation.

  You can use this function to test your implementation of the Named Entity
  Recognition network. When debugging, set max_epochs in the Config object to 1
  so you can rapidly iterate.
  """
  config = Config()
  with tf.Graph().as_default():
    model = SimpleModel(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      for epoch in range(config.max_epochs):
        print('Epoch {}'.format(epoch))
        start = time.time()
        ###
        train_loss, train_acc = model.run_epoch(session, model.X_train,
                                                model.y_train)
        val_loss, predictions = model.predict(session, model.X_dev, model.y_dev)
        print('Training loss: {}'.format(train_loss))
        print('Training acc: {}'.format(train_acc))
        print('Validation loss: {}'.format(val_loss))
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          best_val_epoch = epoch
          if not os.path.exists("./weights"):
            os.makedirs("./weights")
        
          saver.save(session, './weights/ner.weights')
        if epoch - best_val_epoch > config.early_stopping:
          break
        ###
        print('Total time: {}'.format(time.time() - start))
      
      saver.restore(session, './weights/abstract.weights')
      print('Test')
      print('=-=-=')
      print('Writing predictions to abstract.predicted')
      _, predictions = model.predict(session, model.X_test, model.y_test)
      save_predictions(predictions, "abstract.predicted")

if __name__ == "__main__":
  test_model()
