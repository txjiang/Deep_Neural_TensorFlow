from __future__ import print_function
import math
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

def pred(w1, b1, w2, b2, w3, b3, w4, b4, data_set):
    layer1 = tf.nn.relu(tf.matmul(data_set, w1) + b1)
    layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
    layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)
    layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)
    res = tf.nn.softmax(layer4)
    return res

def layer_3_nn_L2_dropout(num_steps=100001, batch_size=128, keep_prob=0.5, beta=0.001):

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
            tf.truncated_normal([image_size * image_size, 1024], stddev=math.sqrt(2.0/(image_size*image_size))))
        biases_input = tf.Variable(
            tf.zeros([1024]))
        weights_layer1 = tf.Variable(
            tf.truncated_normal([1024, 300], stddev=math.sqrt(2.0/1024)))
        biases_layer1 = tf.Variable(
            tf.zeros([300]))
        weights_layer2 = tf.Variable(
            tf.truncated_normal([300, 50], stddev=math.sqrt(2.0/300)))
        biases_layer2 = tf.Variable(
            tf.zeros([50]))
        weights_layer3 = tf.Variable(
            tf.truncated_normal([50, num_labels], stddev=math.sqrt(2.0/50)))
        biases_layer3 = tf.Variable(
            tf.zeros([num_labels]))

        # Training Computation:
        layer1 = tf.matmul(tf_train_dataset, weights_input) + biases_input
        layer1_relu = tf.nn.relu(layer1)
        layer1_dropout = tf.nn.dropout(layer1_relu, keep_prob)

        layer2 = tf.matmul(layer1_dropout, weights_layer1) + biases_layer1
        layer2_relu = tf.nn.relu(layer2)
        layer2_dropout = tf.nn.dropout(layer2_relu, keep_prob)

        layer3 = tf.matmul(layer2_dropout, weights_layer2) + biases_layer2
        layer3_relu = tf.nn.relu(layer3)
        layer3_dropout = tf.nn.dropout(layer3_relu, keep_prob)

        layer_out = tf.matmul(layer3_dropout, weights_layer3) + biases_layer3
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=layer_out)
            + beta * (tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_layer1)
                      + tf.nn.l2_loss(weights_layer2) + tf.nn.l2_loss(weights_layer3)))

        global_step = tf.Variable(0)  # count the number of steps taken.
        learning_rate = tf.train.exponential_decay(0.5, global_step, 10000, 0.96, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # prediction
        train_prediction = pred(weights_input, biases_input, weights_layer1, biases_layer1,
                                weights_layer2, biases_layer2, weights_layer3, biases_layer3, tf_train_dataset)
        valid_prediction = pred(weights_input, biases_input, weights_layer1, biases_layer1,
                                weights_layer2, biases_layer2, weights_layer3, biases_layer3, tf_valid_dataset)
        test_prediction = pred(weights_input, biases_input, weights_layer1, biases_layer1,
                                weights_layer2, biases_layer2, weights_layer3, biases_layer3, tf_test_dataset)

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

layer_3_nn_L2_dropout()