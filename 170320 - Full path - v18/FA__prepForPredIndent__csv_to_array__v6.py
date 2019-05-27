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
# 15.02.17  ekk     Repurposed  Repurposed the heavier prepreation file for training a model into a simple one for
#                               preparing dataset to be predicted the type.
# 15.02.17  ekk     Coppied     Made a copy for the purpose of preparing the "prediction indent"-process.


#IMPORT SOFTWARE
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.fftpack import rfft


def csv_to_array_def(name_feat_csv):
    # Get file
    with open(name_feat_csv, 'r') as f:
        reader = csv.reader(f)
        import_feature_batch_raw = list(reader)
    print("Opened file.")

    # Run rfft() and abs() of features
    import_feature_batch = []
    for i in range(len(import_feature_batch_raw)):
        import_feature_batch.append(abs(rfft(import_feature_batch_raw[i])))
    print("Converted file.")

    # Remove last batch as it may not be a complete 500 feature batch
    # Cretaing boxed result
    prediction_feature_batch = import_feature_batch[:-1]

    print("\nTotal number of batches:      %d" % len(prediction_feature_batch))
    print("Number of features per batch: %d" % len(prediction_feature_batch[0]))
    print("\nInitial setup excecuted succesfully.")

    # Make string to send to GUI
    return_text = "Total number of batches:\t%d" \
                  "\nNumber of features per batch:\t%d" \
                  "\nInitial setup excecuted " \
                  "succesfully." \
                  % (len(prediction_feature_batch), len(prediction_feature_batch[0]))

    # Returning array to be used in TensorFlow and string to GUI
    return prediction_feature_batch
