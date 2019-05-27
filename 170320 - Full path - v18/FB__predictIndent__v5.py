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

# IMPORT SOFTWARE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


def t_interval(mean, standard_deviation, n):
    t_alpha_half = 2.31  # given K.I. of 95%

    lower_value = mean - (t_alpha_half * (standard_deviation / np.sqrt(n)))
    higher_value = mean + (t_alpha_half * (standard_deviation / np.sqrt(n)))

    return_string="\nStudent t confidence interval" \
                  "\nAt a 95 precent cerntainty that the indent is between" \
                  "\n%fmm" \
                  "\nand" \
                  "\n%fmm" % (lower_value, higher_value)

    return return_string


def TF_to_prediction_def(tuplet_packet):
    prediction_feature_batch, prediction_label_batch, model_file_name = tuplet_packet

    # Do not touch, self-regulating values below
    predict_size_x = len(prediction_feature_batch[0]) # Size of X-element
    predict_size_y = 1                                # Size of Y-element
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

    #Needs the model to
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

    # Create a saver.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("Restoring model:  %s" % model_file_name)
        print("...")
        saver.restore(sess, model_file_name)
        print("Model restored.")

        prediction_evaluation_list = y_.eval(feed_dict={x: prediction_feature_batch, keep_prob: 1.0}, session=sess)

    list_sorted = np.sort(prediction_evaluation_list, axis=0)
    fit = stats.norm.pdf(list_sorted, np.mean(list_sorted), np.std(list_sorted))

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.plot(list_sorted, '-o')
    ax1.set_title("Every label sorted by value from low to high")
    ax1.set_xlabel("Every label value in ascending order.") # Source [1]
    ax1.set_ylabel("Amount of indent (mm)")
    ax2 = fig.add_subplot(122)
    ax2.plot(list_sorted, fit, '-o')
    ax2.set_title("Gaussian Distribution")  # Source [2]
    ax2.set_xlabel("Amount of indent (mm)")
    ax2.set_ylabel("Total area under graph equals 1")

    text1 = r'$\mu$ = %fmm' % np.mean(list_sorted)
    text2 = r'$\mu + \sigma$ = %fmm' % (np.mean(list_sorted) + np.std(list_sorted))
    text3 = r'$\mu - \sigma$ = %fmm' % (np.mean(list_sorted) - np.std(list_sorted))

    ax2.axvline(np.mean(list_sorted), color='black', label=text1)
    ax2.axvline(np.mean(list_sorted) + np.std(list_sorted), color='black', label=text2)
    ax2.axvline(np.mean(list_sorted) - np.std(list_sorted), color='black', label=text3)

    # ax2 legend, source [3, 4]
    ax2_ = ax2.legend(handlelength=0, handletextpad=0)
    for item in ax2_.legendHandles:
        item.set_visible(False)

    ax2.annotate(r'$\mu$', xy=(np.mean(list_sorted), 0), xytext=(np.mean(list_sorted), 0),  xycoords='data',
                size=14, ha='center', va='bottom', textcoords='offset pixels',
                bbox=dict(boxstyle='round', fc='w'))

    ax2.annotate(r'$\mu + \sigma$', xy=(np.mean(list_sorted) + np.std(list_sorted), 0), xytext=(np.mean(list_sorted), 0), xycoords='data',
                 size=14, ha='center', va='bottom', textcoords='offset pixels',
                 bbox=dict(boxstyle='round', fc='w'))

    ax2.annotate(r'$\mu - \sigma$', xy=(np.mean(list_sorted) - np.std(list_sorted), 0), xytext=(np.mean(list_sorted), 0), xycoords='data',
                 size=14, ha='center', va='bottom', textcoords='offset pixels',
                 bbox=dict(boxstyle='round', fc='w'))

    plt.show()
    """
    list_average = np.mean(list_sorted)
    print("\nMean:               %f" % list_average)
    print("Standard deviation: %f" % np.std(list_sorted))
    print("\nQ1: (mean - std)    %f" % list_average - np.std(list_sorted))
    print("Q2: (mean)          %f" % (np.mean(list_sorted)))
    print("Q3: (mean + std)    %f" % list_average - np.std(list_sorted))
    find_median = np.median(prediction_evaluation_list)
    print("\nMedian:             %f" % find_median)
    """
    return_string_1 = "Mean:\t\t\t%fmm" % np.mean(list_sorted)
    return_string_2 = "\nStandard deviation:\t%fmm" % np.std(list_sorted)
    return_string_3 = t_interval(np.mean(list_sorted), np.std(list_sorted, np.std(list_sorted), len(prediction_feature_batch)))
    return_string = return_string_1 + return_string_2 + return_string_3

    return return_string


# Sources
# [1] http://matplotlib.org/users/text_intro.html
# [2] http://www.astroml.org/book_figures/chapter3/fig_gaussian_distribution.html
# [3] http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.legend
# [4] http://matplotlib.org/users/legend_guide.html
