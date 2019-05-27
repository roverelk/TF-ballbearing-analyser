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
# 15.02.17  ekk     Edition     Copied the same code as in csv_to_array for training the class-model. It had better
#                               naming conventions. Except from that, the code was identical.


#IMPORT SUB-LIBARY
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.fftpack import rfft
from sklearn.utils import shuffle


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

        # Converting the array with first rfft(), then with abs().
        for i in range(len(import_feat_1r)):
            import_feat_1.append(abs(rfft(import_feat_1r[i])))

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

    info_string = (
    "\nTotal number of feature training batches:\t%d"
    "\nTotal number of label training batches:\t%d"
    "\nTotal number of feature test batches:\t%d"
    "\nTotal number of label test batches:\t\t%d"
    "\n-----------------------------------------------------------------"
    "\nNumber of features per batch:\t\t%d"
    "\nNumber of labels per batch:\t\t%d"
    "\n\nInitial setup excecuted succesfully."
    "\n\nShuffle dataset ..."
    % (len(local_train_feature_array), len(local_train_label_array), len(local_test_feature_array),
       len(local_test_label_array), len(local_train_feature_array[0]), len(local_train_label_array[0])))

    shuffled_train_feature = np.array(local_train_feature_array)
    shuffled_train_label = np.array(local_train_label_array)
    shuffled_test_feature = np.array(local_test_feature_array)
    shuffled_test_label = np.array(local_test_label_array)

    # Shuffle the pairs equally, so the positional values of the feature- and label-array still matches up.
    shuffled_train_feature, shuffled_train_label = shuffle(shuffled_train_feature, shuffled_train_label, random_state=0)
    shuffled_test_feature, shuffled_test_label = shuffle(shuffled_test_feature, shuffled_test_label, random_state=0)

    return shuffled_train_feature, shuffled_train_label, shuffled_test_feature, shuffled_test_label, info_string
