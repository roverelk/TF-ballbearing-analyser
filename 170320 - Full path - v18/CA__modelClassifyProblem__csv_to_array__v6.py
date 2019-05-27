# PURPOSE: Import own .csv-created batches and prepering them in an array for TensorFlow
#
# LOG
# Date------Init----What--------Comment------------------------------------
# 24.01.17  ekk     Created
# 24.01.17  ekk     Comment     Deler opp programmet så det er mulig å se på hver enkelt del.
# 30.01.17  ekk     Cleaned     Koden tar nå bare inn ett sett med .csv
# 08.02.17  ekk     Multiple    Re-arranging so it will use multiple input at the same time again. So to train the model
#                               with multiple inputs at once. So not to climatize to a single input at a time.
# 08.02.17  ekk     Shuffle     This program now shuffles the output arrays in unison so the corresponding x- and y-
#                               values have the same position in their respective arrays.
# 14.02.17  ekk     Cleaning    Cleaning the script to fit with a better structure of the whole program.
# 14.02.17  ekk     Comment     Adding comments so to make it easier to read the code.
# 28.02.17  ekk     GUI         Connects to GUI instead of to main
# 15.03.17  ekk     Cleaning    Working over the language making it consitant and readable for outsiders.


#IMPORT SOFTWARE
import numpy as np
import csv
import matplotlib.pyplot as plt

from scipy.fftpack import rfft
from scipy import arange, fft
from sklearn.utils import shuffle
from pylab import plot, show, title, xlabel, ylabel, subplot
from numpy import sin, linspace, pi


def csv_to_array_def(input_feature_array, input_label_array, percent_train):

    local_train_feature_array = []
    local_train_label_array = []
    local_test_feature_array = []
    local_test_label_array = []



    for j in range(len(input_feature_array)):
        # Getting .csv-file number j in array "input_feature_array".
        get_file = input_feature_array[j]
        # Reading .csv-file into an array
        with open(get_file, 'r') as f:
            reader = csv.reader(f)
            import_feat_1r = list(reader)
        import_feat_1 = []
        # Convering the array with first rfft(), then with abs().
        for i in range(len(import_feat_1r)):
            import_feat_1.append(abs(rfft(import_feat_1r[i])))


        Fs = 12000.0  # Sampling rate
        Ts = 1.0 / Fs  # sampling interval

        y = import_feat_1r[0]  # Signal
        n = len(import_feat_1r[0])  # Length of the signal
        k = arange(n)
        T = n / Fs

        frq = k / T   # two sides frequency range
        frq = frq[:-np.int(n / 2)]  # one side frequency range

        print("Frq: ", frq)
        Y = rfft(y) / n  # fft computing and normalization
        Y = Y[:-np.int(n / 2)]

        t = []
        for i in range(len(y)):
            t.append(Ts*i)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        fig.subplots_adjust(top=0.85)
        ax1.set_title('Time-domain')
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('Amplitude')
        plt.plot(t, y, 'r')
        ax2 = fig.add_subplot(212)
        ax2.set_title('Frequency-domain')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('|Y(freq)|')
        ax2.plot(frq, abs(Y), 'r')
        plt.show()


        # Splitting the array into a train and a test part.Based on "percent_train".
        # Cutting away the last position in the array to remove the possibility of an unfinished array.
        number_of_train_feat_1 = int(percent_train * len(import_feat_1))
        train_feat_1 = import_feat_1[:number_of_train_feat_1]
        test_feat_1 = import_feat_1[number_of_train_feat_1:-1]

        # Create the label-arrays with fixed values of similar lengths to the equivalent feature-arrays.
        train_lab_1 = []
        for i in range(len(train_feat_1)):
            train_lab_1.append(input_label_array[j])
        test_lab_1 = []
        for i in range(len(test_feat_1)):
            test_lab_1.append(input_label_array[j])

        # Adding the processed arrays to an overall array featuring every input-file.
        # PS: Important to stack feature- and label-arrays corresponding positional values.
        local_train_feature_array = local_train_feature_array + train_feat_1
        local_train_label_array   = local_train_label_array   + train_lab_1
        local_test_feature_array  = local_test_feature_array  + test_feat_1
        local_test_label_array    = local_test_label_array    + test_lab_1

    print("\nTotal number of feature training batches: %d" % len(local_train_feature_array))
    print("Total number of label training batches:   %d" % len(local_train_label_array))
    print("Total number of feature test batches:     %d" % len(local_test_feature_array))
    print("Total number of label test batches:       %d" % len(local_test_label_array))
    print("---------------------------------------------")
    print("Number of features per batch:             %d" % len(local_train_feature_array[0]))
    print("Number of labels per batch:               %d" % len(local_train_label_array[0]))
    print("\nInitial setup excecuted succesfully.")

    print("\nShuffle dataset ...")

    # Creating a string for output of all values to GUI.
    info_string = ("\nTotal number of feature training batches:\t%d"
                   "\nTotal number of label training batches:"
                   "\t%d\nTotal number of feature test batches:\t%d"
                   "\nTotal number of label test batches:\t\t%d"
                   "\n-----------------------------------------------------------------"
                   "\nNumber of features per batch:\t\t%d"
                   "\nNumber of labels per batch:\t\t%d"
                   "\n\nInitial setup excecuted succesfully."
                   "\n\nShuffle dataset ..."
                   % (len(local_train_feature_array), len(local_train_label_array), len(local_test_feature_array),
                      len(local_test_label_array),len(local_train_feature_array[0]),len(local_train_label_array[0])) )

    # Convert to prepare for shuffling
    shuffled_train_feature = np.array(local_train_feature_array)
    shuffled_train_label = np.array(local_train_label_array)
    shuffled_test_feature = np.array(local_test_feature_array)
    shuffled_test_label = np.array(local_test_label_array)

    # Shuffle the pairs equally, so the positional values of the feature- and label-array still matches up.
    shuffled_train_feature, shuffled_train_label = shuffle(shuffled_train_feature, shuffled_train_label, random_state=0)
    shuffled_test_feature, shuffled_test_label = shuffle(shuffled_test_feature, shuffled_test_label, random_state=0)

    return shuffled_train_feature, shuffled_train_label, shuffled_test_feature, shuffled_test_label, info_string
