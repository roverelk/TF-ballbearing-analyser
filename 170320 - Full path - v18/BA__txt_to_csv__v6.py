# PURPOSE: Convert from a file of numvers in .txt-format. Rewrite into a .csv with a set amount of values per
#           collumns.
#
# LOG
# Date------Init----What--------Comment-------------------------------------------
# 11.01.17  ekk     Created
# 06.02.17  ekk     Comment     Implemented the program into the main.py mainframe.
# 28.02.17  ekk     GUI         Connected to GUI

import csv
import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import *
from PyQt4.QtGui  import *


def txt_to_csv_def(number_of_features, name_input_txt, name_output_csv):

    try:
        importer = []
        nyImporter = []
        u = 0

        with open(name_input_txt) as inputfile:
            for line in inputfile:
                importer.append(line.strip().split())

                nyImporter = nyImporter + importer[u]
                u = u + 1
                QApplication.processEvents()
                print('Number of imported values: %d' % u)


        inputfile.close()
        data = np.asarray(nyImporter)
        chunks = [data[x:x + number_of_features] for x in range(0, len(data), number_of_features)]
        print("Chunks created.")
        # Remove progressbar and label when prosess is done

        with open(name_output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(chunks)

        return "Converstion of text to csv ran successfully!\n\nYou converted:\n%s\nto:\n%s\n\nWith %d  fetures per line." \
               % (name_input_txt, name_output_csv, number_of_features)

    except RuntimeError:
        return "Convertion of text to csv failed to run."
