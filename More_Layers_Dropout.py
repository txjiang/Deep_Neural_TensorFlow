# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])



def logic_L2_reg(num_steps = 3001, batch_size = 128, beta = 0.01):

    graph = tf.Graph()

    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        #logits_train = logits + beta * tf.nn.l2_loss(weights)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
            + beta * tf.nn.l2_loss(weights))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print(
            "-------------------------------------------------------------------------------------------------------------")
        print('Logic Unit-SGD-L2 Regularization'
              '')
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

#logic_L2_reg()
def nn_L2_reg(num_steps = 3001, batch_size = 128, beta = 0.01):

    graph = tf.Graph()

    with graph.as_default():
        # using placeholder, because of SGD, everytime the batch changes
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # variables
        # input->hidden layer 1
        weights_input = tf.Variable(
            tf.truncated_normal([image_size * image_size, 1024]))
        biases_input = tf.Variable(
            tf.zeros([1024]))
        weights_layer1 = tf.Variable(
            tf.truncated_normal([1024, num_labels]))
        biases_layer2 = tf.Variable(
            tf.zeros([num_labels]))

        # Training Computation:
        layer1 = tf.matmul(tf_train_dataset, weights_input) + biases_input
        layer1_relu = tf.nn.relu(layer1)
        layer2 = tf.matmul(layer1_relu, weights_layer1) + biases_layer2
        #logits = layer2 + beta * (tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_layer1))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=layer2)
             + beta * (tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_layer1)))

        # Optimizer:
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # prediction
        train_prediction = tf.nn.softmax(layer2)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_input) + biases_input),
                      weights_layer1) + biases_layer2)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_input) + biases_input),
                      weights_layer1) + biases_layer2)

    with tf.Session(graph=graph) as session:
        print(
            '-------------------------------------------------------------------------------------------------------------')
        print('Single Hidden Layer (Relu) NN')
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

def nn_L2_dropout(num_steps=3001, batch_size=128, keep_prob=0.5, beta=0.01):

    graph = tf.Graph()

    with graph.as_default():
        # using placeholder, because of SGD, everytime the batch changes
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # variables
        # input->hidden layer 1
        weights_input = tf.Variable(
            tf.truncated_normal([image_size * image_size, 1024]))
        biases_input = tf.Variable(
            tf.zeros([1024]))
        weights_layer1 = tf.Variable(
            tf.truncated_normal([1024, num_labels]))
        biases_layer2 = tf.Variable(
            tf.zeros([num_labels]))

        # Training Computation:
        layer1 = tf.matmul(tf_train_dataset, weights_input) + biases_input
        #print('debug')
        #print(tf_train_dataset.shape)
        layer1_relu = tf.nn.relu(layer1)
        #print('debug')
        #print(layer1_relu.shape)
        layer1_dropout = tf.nn.dropout(layer1_relu, keep_prob)
        #print(layer1_dropout.shape)
        layer2 = tf.matmul(layer1_dropout, weights_layer1) + biases_layer2
        #print(layer2.shape)
        #logits = layer2 + beta * (tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_layer1))
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=layer2)
            + beta * (tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_layer1)))

        # Optimizer:
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # prediction
        train_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_train_dataset, weights_input) + biases_input),
                      weights_layer1) + biases_layer2)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_input) + biases_input),
                      weights_layer1) + biases_layer2)
        test_prediction = tf.nn.softmax(
            tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_input) + biases_input),
                      weights_layer1) + biases_layer2)

    with tf.Session(graph=graph) as session:
        print(
            '-------------------------------------------------------------------------------------------------------------')
        print('Single Hidden Layer (Relu) NN---With Dropout')
        tf.global_variables_initializer().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
#logic_L2_reg()
#nn_L2_reg()
nn_L2_dropout()