# PURPOSE: Create TensorFlow model to predict type of error.
#
# LOG
# Date------Init----What-------------Comment----------------------------------
# 04.01.17  ekk     Opprettet       Source: https://www.tensorflow.org/versions/master/tutorials/mnist/pros/
# 04.01.17  ekk     Kommentar       Programmet er tregt på min PC. Programmet gir ut resultater undervei mens det
#                                   kjører.
# 24.01.17  ekk     Kommentar       Fått ny innsikt og skal prøve å forandre det til et brukbart program.
# 25.01.17  ekk     Kommentar       Dette programmet virker BEST PÅ GPU!
# 26.01.17  ekk     Kommentar       Dette er nå et delprogram som kjørers fra "main.py", det kan ikke lengre operes av
#                                   seg selv.
# 29.01.17  ekk     Lagre data      Måtte legge til "import os", og de delene som henter fra "os" for å kunne lagre.
# 06.02.17  ekk     Save model      Source: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
#                                   Uses TF's e.g. for saving the model.
# 07.02.17  ekk     Restore model   Source: https://nathanbrixius.wordpress.com/2016/05/24/checkpointing-and-reusing-tensorflow-models/
#                                   Been able to save and restore a TF-model.
# 14.02.17  ekk     Cleaning        Cleaning the script to fit with a better structure of the whole program.

# IMPORT SOFTWARE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import *
from PyQt4.QtGui  import *


def save(checkpoint_file):
    with tf.Session() as session:
        print(session.run(tf.all_variables()))
        saver = tf.train.Saver([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
        saver.save(session, checkpoint_file)

def restore(checkpoint_file):
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_file)
        print(session.run(tf.initialize_all_variables()))

def reset():
    tf.reset_default_graph()


def array_to_TF_def(tuplet_input):
    iteration_number = 0

    train_feature_batch, train_label_batch, test_feature_batch, test_label_batch, gradient_length, \
    print_intermidiate_values_train, print_intermidiate_values_test, print_error_rate, print_graph, fil_navn, \
    iteration_number, fil_ny = tuplet_input

    #Creating the get-file string
    checkpoint_file = "./" + fil_navn + "-" + str(iteration_number)
    saver_file = "./" + fil_navn

    # Do not touch, self-regulating values below
    train_iterations = len(train_feature_batch)    # Number of iterations taken by TensorFlow
    train_size_x = len(train_feature_batch[0])     # Size of feature element
    train_size_y = len(train_label_batch[0])      # Size of label element
    #######################

    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, shape=[None, train_size_x])
    y = tf.placeholder(tf.float32, shape=[None, train_size_y])

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(x, [-1, 25, 20, 1])

    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])

    # Reduces the 25x20 to a 14x10 matrix
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])

    # Reduces the 14x10 to a 7x5 matrix
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Add a fully connected layer
    W_fc1 = weight_variable([7 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, train_size_y])
    b_fc2 = bias_variable([train_size_y])

    y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # Create a saver.
    saver = tf.train.Saver()

    # Define the training steps and rate of adjustment to every W and b with AdamOptimizer.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
    train_step = tf.train.AdamOptimizer(gradient_length).minimize(cross_entropy)

    # Self evaluation of learning process while it runs.
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialise all variables
    sess.run(tf.global_variables_initializer())

    batch_feat = np.asarray(train_feature_batch)
    batch_lab = np.asarray(train_label_batch)

    # Create a array to save progress for accuracy for the learning
    accuracy_plot_save = []

    # Control if we are continuing to teach an old model or creating a new one.
    if fil_ny:
        reset()
        print("Loading %s-%.0f ..." % (fil_navn, iteration_number))
        saver.restore(sess, checkpoint_file)
        sess.run([W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2])
        print("File loaded.")

    # This def is only make the numbers stack correctly in system print. It has nothing to do with TensorFlow or
    # the output string being sent to the GUI.
    def set_number_of_zeros(newBatch_feat_, i_):
        k_int = 0
        k_str = ""
        for j in range(len(str(len(newBatch_feat_))) - len(str(i_))):
            k_int += 1
        while k_int > 0:
            k_str += "0"
            k_int -= 1
        return k_str

    for i in range(train_iterations):
        # Converting format for training
        batch_feat[i] = np.array(batch_feat[i], np.float32).reshape(1, len(batch_feat[i]))
        batch_lab[i]  = np.array(batch_lab[i], np.float32).reshape(1, len(batch_lab[i]))

        if i % 10 == 0:
            k_str = set_number_of_zeros(train_feature_batch, i)
            train_accuracy = accuracy.eval(feed_dict={x: batch_feat, y: batch_lab, keep_prob: 1.0})
            print("Step %s%d/%d, training accuracy: %g" % (k_str, i, len(train_feature_batch), train_accuracy))
            accuracy_plot_save.append(train_accuracy)
        if i % 50 == 0:
            saver.save(sess, saver_file, global_step=(i+iteration_number))
        train_step.run(feed_dict={x: batch_feat, y: batch_lab, keep_prob: 0.5})
        QApplication.processEvents()

    # Convert shape for testing
    test_feature_batch_shaped = np.asarray(test_feature_batch)
    test_label_batch_shaped = np.asarray(test_label_batch)
    for i in range(len(test_feature_batch_shaped)):
        test_feature_batch_shaped[i] = np.array(test_feature_batch_shaped[i], np.float32).reshape(1, len(test_feature_batch_shaped[0]))
        test_label_batch_shaped[i] = np.array(test_label_batch_shaped[i], np.float32).reshape(1, len(test_label_batch_shaped[0]))

    print("TEST ACCURACY: %g" % accuracy.eval(feed_dict={x: test_feature_batch_shaped, y: test_label_batch_shaped, keep_prob: 1.0}))

    # Show plot over the training process
    plt.plot([np.mean(accuracy_plot_save[i]) for i in range(len(accuracy_plot_save))])
    plt.show()

    # Return string to show to user in GUI
    return("TEST ACCURACY:\t%g" % accuracy.eval(feed_dict={x: test_feature_batch_shaped, y: test_label_batch_shaped, keep_prob: 1.0}))


