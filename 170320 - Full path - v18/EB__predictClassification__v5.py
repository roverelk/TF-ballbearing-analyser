# HENSIKT:  Importere egne tall inn til programmet.
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
# 07.02.17 ekk      Restore model   Source: https://nathanbrixius.wordpress.com/2016/05/24/checkpointing-and-reusing-tensorflow-models/
#                                   Been able to save and restore a TF-model.
# 16.03.17  ekk     Cleaning        Making it able for others to read the code aswell.

# IMPORT SOFTWARE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.fftpack import rfft
import os

def TF_to_prediction_def(tuplet_packet):
    prediction_feature_batch, prediction_label_batch, model_file_name = tuplet_packet

    # Do not touch, self-regulating values below
    predict_size_x = len(prediction_feature_batch[0]) # Size of X-element
    predict_size_y = len(prediction_label_batch[0])  # Size of Y-element
    #######################

    x = tf.placeholder(tf.float32, shape=[None, predict_size_x])

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

    # The same model as in training to set up the right structure
    x_vibration_shaped = tf.reshape(x, [-1, 25, 20, 1])
    W_conv1 = weight_variable([5, 5, 1, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_vibration_shaped, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([7 * 5 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 5 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, predict_size_y])
    b_fc2 = bias_variable([predict_size_y])
    y_ = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    init_op = tf.global_variables_initializer()
    prediction_evaluation_list = []

    # Create a saver. Also need to get a trained model.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("Restoring model:  %s" % model_file_name_)
        print("...")
        saver.restore(sess, model_file_name)
        print("Model restored.")

        prediction = tf.argmax(y_, 1)
        prediction_evaluation_list = prediction.eval(feed_dict={x: prediction_feature_batch, keep_prob: 1.0}, session=sess)

    # Creat an array to count the 4 possible answers.
    probability_counter = []
    for i in range(4):
        probability_counter.append(0)

    # Count up how many times the model predicted different states.
    for i in range(len(prediction_evaluation_list)):
        probability_counter[prediction_evaluation_list[i]] = probability_counter[prediction_evaluation_list[i]] + 1

    # For easy use by user create a percentage answere.
    baseline_percent  = 100 * (probability_counter[0]) / len(prediction_evaluation_list)
    outerring_percent = 100 * (probability_counter[1]) / len(prediction_evaluation_list)
    innerring_percent = 100 * (probability_counter[2]) / len(prediction_evaluation_list)
    ball_percent      = 100 * (probability_counter[3]) / len(prediction_evaluation_list)

    # The status text for system.out
    print("\nPROBABILITY")
    print("Out of %f batches of features is predicted that:", sum(probability_counter))
    print("Baseline:   %f\tor\t%f percent" % (probability_counter[0], baseline_percent))
    print("Outer ring: %f\tor\t%f percent" % (probability_counter[1], outerring_percent))
    print("Inner ring: %f\tor\t%f percent" % (probability_counter[2], innerring_percent))
    print("Ball:       %f\tor\t%f percent" % (probability_counter[3], ball_percent))

    # The same status just written to be sent to the GUI.
    return_string = "\nPROBABILITY:" \
                    "\nOut of %f batches of features is predicted that" \
                    "\nBaseline:\t%f\tor\t%f percent" \
                    "\nOuter ring:\t%f\tor\t%f percent" \
                    "\nInner ring:\t%f\tor\t%f percent" \
                    "\nBall:\t\t%f\tor\t%f percent" \
                    % (sum(probability_counter), probability_counter[0], baseline_percent, probability_counter[1],
                       outerring_percent, probability_counter[2], innerring_percent, probability_counter[3], ball_percent)

    return return_string
