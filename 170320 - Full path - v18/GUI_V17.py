import sys
import os
from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QApplication
from PyQt4.QtCore import *
from PyQt4.QtGui  import *


from PyQt4.QtCore import pyqtSlot

from BA__txt_to_csv__v6                         import txt_to_csv_def       as convert_txt_to_csv
from CA__modelClassifyProblem__csv_to_array__v6 import csv_to_array_def     as import_csv_for_classification
from CB__TF_trainModel__v10                     import array_to_TF_def      as train_model_classification
from DA__modelIndentSize__csv_to_array__v6      import csv_to_array_def     as import_csv_for_indent
from DB__TF_trainModel__v10                     import array_to_TF_def      as train_model_indent
from EA__prepForPredClass__csv_to_array__v6     import csv_to_array_def     as import_csv_to_predict_classification
from EB__predictClassification__v5              import TF_to_prediction_def as predict_classification
from FA__prepForPredIndent__csv_to_array__v6    import csv_to_array_def     as import_csv_to_predict_indent
from FB__predictIndent__v5                      import TF_to_prediction_def as predict_indent

#Might move this script to it's own file.
from CB__TF_trainModel__v10 import reset as reset_tf



class Window(QtGui.QMainWindow):
    def __init__(self):
        #Starting the main window (QtGui.QMainWindow)
        super(Window, self).__init__()

        # Setting the geometry of the window and moving it to a fixed area when started
        self.setGeometry(400, 200, 1200, 800)

        # Locking the size to (1200, 800)
        self.setFixedSize(1200,800)

        # Setting the title of the window
        self.setWindowTitle("Tensorflow")

        # Setting the GUI icon
        self.setWindowIcon(QtGui.QIcon('tf.jpg'))

        # Setting the style of the GUI. Other alternatives: Plastique, Windows and more
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create("CleanLooks"))

        # Creating a exit function
        extractAction = QtGui.QAction("&Exit", self)

        # Setting a exit shortcut
        extractAction.setShortcut("Ctrl+Q")

        # Close the GUI if shortcut is used
        extractAction.triggered.connect(self.close_application)

        # Creating a info box
        openAbout = QtGui.QAction("&About", self)

        # Setting a shortcut
        openAbout.setShortcut("Ctrl+E")
        #openAbout.setStatusTip('About')

        # Open about window is shortcut is used
        openAbout.triggered.connect(self.open_About)

        # Code for picture, can be used if needed
        """picture = QtGui.QPixmap("tf.jpg")
        picture_label = QtGui.QLabel(self)
        picture_label.setPixmap(picture)
        picture_label.setVisible(True)
        picture_label.move(400, 200)"""

        # Creating a GIF and adding a GIF file
        bearing_gif = QtGui.QMovie("BallBearing.gif")

        # Creating a label
        gif_label = QtGui.QLabel(self)

        # Adding the GIF to a label
        gif_label.setMovie(bearing_gif)

        # Setting the size of the GIF
        gif_label.setGeometry(700, 200, 500, 300)

        # Starting the Gif
        bearing_gif.start()

        # Creating the menubar and adding a filemenu
        mainMenu = self.menuBar()

        # Adding 'File' to the menu
        fileMenu = mainMenu.addMenu('&File')

        # Connecting function to 'file'
        fileMenu.addAction(extractAction)

        # Adding 'About' to menu
        AboutMenu = mainMenu.addMenu("&About")

        # Connecting function to 'About'
        AboutMenu.addAction(openAbout)

        # Starting the home function that creates buttons, texteditor, toolbar and progress bar
        self.home()

    def home(self):

        # Creating a textfield for output
        self.output_field = QtGui.QTextEdit(self)

        # Disabling the textfield for user input, making it a read only
        self.output_field.setReadOnly(True)

        # Creating a font
        font = QtGui.QFont()

        # Setting the font size to 10
        font.setPointSize(10)

        # Adding the font size to the textedit field
        self.output_field.setFont(font)

        # Setting the intro text
        self.output_field.setText("Welcome to this application!")
        self.output_field.append("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nNote:"
                                 "\nIf you try to run the training process without using the GPU version of Tensorflow "
                                 "enabled, it might take several hours...")

        # Moving the textfield
        self.output_field.move(100, 100)

        # Setting the textfield size
        self.output_field.resize(500, 500)

        # Creating an exit function to tie to toolbar
        extractAction = QtGui.QAction(QtGui.QIcon('tf.jpg'), 'Exit', self)
        extractAction.triggered.connect(self.close_application)

        # Creating a toolbar
        self.toolBar = self.addToolBar("Exit")
        self.toolBar.addAction(extractAction)

        # Creating buttons
        self.predict_btn = QtGui.QPushButton("Predict", self)
        self.trainModel_btn = QtGui.QPushButton("Train Model", self)
        self.convert_btn = QtGui.QPushButton("Convert", self)
        self.quit_btn = QtGui.QPushButton("Quit", self)

        # Set fixed size to buttons
        self.trainModel_btn.setFixedSize(100, 30)
        self.predict_btn.setFixedSize(100,30)
        self.convert_btn.setFixedSize(100, 30)
        self.quit_btn.setFixedSize(100, 30)

        # Move buttons
        self.predict_btn.move(575, 772)
        self.trainModel_btn.move(475, 772)
        self.convert_btn.move(375, 772)
        self.quit_btn.move(675, 772)

        # Adding functions to buttons
        self.quit_btn.clicked.connect(self.close_application)
        self.predict_btn.clicked.connect(self.predict_model)
        self.convert_btn.clicked.connect(self.convert_clicked)
        self.trainModel_btn.clicked.connect(self.train_model_clicked)

        # Creating a progressbar to indicate that a process is running
        self.progress = QtGui.QProgressBar(self)

        # Set max and min to zero: this make the progressbar run even though it doesnt know the lenght of the process
        self.progress.setMaximum(0)
        self.progress.setMinimum(0)

        # Set processbar size
        self.progress.setGeometry(350, 650, 500, 55)

        # Hide progress bar until it is needed
        self.progress.hide()

        # show all the buttons, gif and textedit fields that has been created in this function
        self.show()

    def buttons_On(self):
        #Method to turn on buttons after they have been disabled
        self.convert_btn.setEnabled(True)
        self.trainModel_btn.setEnabled(True)
        self.predict_btn.setEnabled(True)

    def buttons_Off(self):
        # Method to turn off buttons during a process
        self.convert_btn.setEnabled(False)
        self.trainModel_btn.setEnabled(False)
        self.predict_btn.setEnabled(False)

    def open_About(self):
        # Opening messagebox with information
        choice = QtGui.QMessageBox.question(self, 'About',
                                            "Still under testing"
                                            "\nDesigned by: Fredrik Kvist, Eirik K. Kjevik and Silja Svensson",
                                            QtGui.QMessageBox.Ok)
        if choice == QtGui.QMessageBox.Ok:
            pass
        else:
            pass

    def train_model_clicked(self):
        # .ckpt-fil

        file_name = "model_fault_inner_v1.ckpt"
        iteration_number = 0
        self.output_field.setText("Initilizing training...")

        new_train_feature = []
        new_train_label = []
        new_test_feature = []
        new_test_label = []
        batch_split_percent = 0.85
        gradient_length = 0.000001  # Change to change to the learningsteps

        class_feature = []
        class_label = []

        class_print_intermidiate_values_train = False
        class_print_intermidiate_values_test = False
        class_print_error_rate = True
        class_print_graph = True
        class_new_file = False  # False = create new file
                                # True  = load old file with name 'file_name'

        indent_feature = []
        indent_label = []

        indent_print_intermidiate_values_train = False
        indent_print_intermidiate_values_test = False
        indent_print_error_rate = True
        indent_print_graph = True
        indent_new_file = False  # False = create new file
                                 # True  = load old file with name 'file_name'

        # Creating buttons
        button_classification = QtGui.QPushButton("Classification")
        button_indent = QtGui.QPushButton("Indentation")
        button_cancel_model = QtGui.QPushButton("Cancel")
        button_inner = QtGui.QPushButton("Innerring")
        button_outer = QtGui.QPushButton("Outerring")
        button_ball = QtGui.QPushButton("Ball")
        button_base = QtGui.QPushButton("No damage")
        button_cancel = QtGui.QPushButton("Cancel")
        button_yes = QtGui.QPushButton("Yes")
        button_no = QtGui.QPushButton("No")

        # Creating Messageboxes for user input
        choose_model_msgBox = QtGui.QMessageBox()
        choose_model_msgBox.setWindowTitle("Choose model")
        choose_model_msgBox.setIcon(QMessageBox.Question)
        choose_model_msgBox.setText('Which model would you like to train?')

        choose_model_msgBox.setDetailedText("To define the size of the damage to de ball bearing it is necessary to "
                                            "use two models. "
                                            "Firstly the classification model to determine if there is "
                                            "a damage, and which type it is most likely to be. "
                                            "Secondly the indentation model to determine the size of the "
                                            "indentation.")

        class_msgBox = QtGui.QMessageBox()
        class_msgBox.setText('Where is the damage located?')
        class_msgBox.setIcon(QMessageBox.Question)

        class_msgBox.setDetailedText("To train a model each sets of features in csv-format needs to add a label so to "
                                     "know what the correct results of the training. Each csv-file added to the"
                                     "training of the model needs a label.\n\nIt is necessary to add data sets of all "
                                     "the types of cases the model should be able to predict.")

        class_msgBox.setWindowTitle("Classification model")
        add_files_msgBox = QtGui.QMessageBox()
        add_files_msgBox.setWindowTitle("Add files")
        add_files_msgBox.setIcon(QMessageBox.Question)
        add_files_msgBox.setText('Do you wish to add another set of files')

        add_files_msgBox.setDetailedText("To train a model that are able to recognise every type of damage in a ball "
                                         "bearing it will be necessary to at least supply one data set representing "
                                         "each type of damage, and one to represent the no-damage baseline.")

        # Adding icon
        class_msgBox.setWindowIcon(QtGui.QIcon('tf.jpg'))
        choose_model_msgBox.setWindowIcon(QtGui.QIcon('tf.jpg'))
        add_files_msgBox.setWindowIcon(QtGui.QIcon('tf.jpg'))

        # Adding buttons to the different messageboxes
        class_msgBox.addButton(button_inner, QtGui.QMessageBox.YesRole)
        class_msgBox.addButton(button_outer, QtGui.QMessageBox.YesRole)
        class_msgBox.addButton(button_ball, QtGui.QMessageBox.YesRole)
        class_msgBox.addButton(button_base, QtGui.QMessageBox.YesRole)
        class_msgBox.addButton(button_cancel, QtGui.QMessageBox.RejectRole)
        add_files_msgBox.addButton(button_yes, QtGui.QMessageBox.YesRole)
        add_files_msgBox.addButton(button_no, QtGui.QMessageBox.YesRole)
        choose_model_msgBox.addButton(button_classification, QtGui.QMessageBox.YesRole)
        choose_model_msgBox.addButton(button_indent, QtGui.QMessageBox.YesRole)
        choose_model_msgBox.addButton(button_cancel_model, QtGui.QMessageBox.YesRole)

        # Starting the messagebox
        choose_model_msgBox.exec_()
        # Try/Except in case of errors
        try:
            self.buttons_Off()
            if choose_model_msgBox.clickedButton() == button_cancel_model:
                self.output_field.setText("Welcome to this application\n\nCancelled")
                self.buttons_On()

            if choose_model_msgBox.clickedButton() == button_classification:
                continue_append = True
                self.output_field.append("Classification model chosen")

                while continue_append:

                    file = QtGui.QFileDialog.getOpenFileName(self, "Open CSV file", " ", "*.csv")
                    if file is "":
                        self.buttons_On()
                        break
                    else:
                        class_feature.append(file)
                    # initize  classification messagebox
                    class_msgBox.exec_()
                    if class_msgBox.clickedButton() == button_inner:
                        class_label.append([0, 0, 1, 0])
                    if class_msgBox.clickedButton() == button_outer:
                        class_label.append([0, 1, 0, 0])
                    if class_msgBox.clickedButton() == button_ball:
                        class_label.append([0, 0, 0, 1])
                    if class_msgBox.clickedButton() == button_base:
                        class_label.append([1, 0, 0, 0])

                    if class_msgBox.clickedButton() == button_cancel:
                        self.progress.hide()
                        self.output_field.setText("Welcome to this application\n\nCancelled")
                        # Cancel process and break from while loop
                        break
                    # initize add files messagebox
                    add_files_msgBox.exec_()
                    if add_files_msgBox.clickedButton() == button_no:
                        continue_append = False
                # show the progressbar
                self.progress.show()
                new_train_feature, new_train_label, new_test_feature, new_test_label, info_string = \
                    import_csv_for_classification(class_feature, class_label, batch_split_percent)
                # Showing what parameters are being used
                self.output_field.append(info_string)

                class_train_tuplet = new_train_feature, new_train_label, new_test_feature, new_test_label, gradient_length, \
                                     class_print_intermidiate_values_train, class_print_intermidiate_values_test, \
                                     class_print_error_rate, class_print_graph, file_name, iteration_number, \
                                     class_new_file
                reset_tf()
                self.output_field.append("Training process started:"
                                         "\nthis can take minutes or hours depending on your setup")
                show_result = train_model_classification(class_train_tuplet)
                self.output_field.append(show_result)
                self.buttons_On()
                self.progress.hide()

            elif choose_model_msgBox.clickedButton() == button_indent:
                continue_append = True
                self.output_field.append("Indentation model chosen")

                # While method to add more files to the training process
                while continue_append:

                    file = QtGui.QFileDialog.getOpenFileName(self, "Open CSV file", " ", "*.csv")
                    if file is "":
                        self.buttons_On()
                        break
                    else:
                        indent_feature.append(file)
                    num, ok = QInputDialog.getDouble(self, "Damage", "Enter indent in millimeters", 0.0000, 0 ,100 , 4)

                    if ok:
                        indent_label.append([num])
                        # intiate add files messagebox
                        add_files_msgBox.exec_()
                    else:
                        break

                    if add_files_msgBox.clickedButton() == button_no:
                        continue_append = False

                # Show the progressbar
                self.progress.show()

                new_train_feature, new_train_label, new_test_feature, new_test_label, info_string = import_csv_for_indent(indent_feature, indent_label, batch_split_percent)

                self.output_field.append(info_string)
                self.output_field.append("Training process started:\nthis can take minutes or hours depending on your setup")

                indent_train_tuplet = new_train_feature, new_train_label, new_test_feature, new_test_label, gradient_length, indent_print_intermidiate_values_train, \
                                      indent_print_intermidiate_values_test, indent_print_error_rate, indent_print_graph, file_name, iteration_number, indent_new_file
                reset_tf()

                show_result = train_model_indent(indent_train_tuplet)

                # Hide progressbar, show result of training and enable buttons
                self.output_field.append(show_result)

                self.progress.hide()
                self.buttons_On()

        except:
            # Exception that activates the buttons, hides the progressbar and resets the output text
            self.buttons_On()
            self.output_field.setText("Welcome to this application.\n\nCancelled")
            self.progress.hide()

    def predict_model(self):

        # Creating messagebox for input
        predict_model_msgBox = QtGui.QMessageBox()
        predict_model_msgBox.setWindowTitle("Choose model")
        predict_model_msgBox.setIcon(QMessageBox.Question)
        predict_model_msgBox.setText('Which model would you like to prediction?')

        # Creating buttons
        predict_classification_btn = QtGui.QPushButton("Classification")
        predict_indent_btn = QtGui.QPushButton("Indentation")
        predict_cancel_model_btn = QtGui.QPushButton("Cancel")

        # Adding the buttons to predict_model_msgbox
        predict_model_msgBox.addButton(predict_classification_btn, QtGui.QMessageBox.YesRole)
        predict_model_msgBox.addButton(predict_indent_btn, QtGui.QMessageBox.YesRole)
        predict_model_msgBox.addButton(predict_cancel_model_btn, QtGui.QMessageBox.YesRole)
        predict_model_msgBox.setDetailedText("It is necessary to already have trained a model to use one.")

        # Set icon
        predict_model_msgBox.setWindowIcon(QtGui.QIcon('tf.jpg'))

        # Try/Except in case of errors
        try:
            self.buttons_Off()
            self.output_field.setText("Starting prediction\n\nPlease select CKPT file and CSV file for prediction")
            predict_model_msgBox.exec_()
            if predict_model_msgBox.clickedButton() == predict_classification_btn:

                class_model_name = QtGui.QFileDialog.getOpenFileName(self, 'Open CKPT File' , '' ,
                                                                     '(*.meta)(*.index)(*.data-00000-of-00001)')
                # Redudancy: Is covered by class_model_name QFileDialog
                if class_model_name is "":
                    self.buttons_On()
                    raise Exception
                elif class_model_name.endswith('.meta'):
                    class_model_name = class_model_name[:-5]
                elif class_model_name.endswith('.index'):
                    class_model_name = class_model_name[:-6]
                elif class_model_name.endswith('.data-00000-of-00001'):
                    class_model_name = class_model_name[:-20]
                print(class_model_name)
                class_pred_file = QtGui.QFileDialog.getOpenFileName(self, 'Open CSV File', '', '(*.csv)')
                if class_pred_file is "":
                    self.buttons_On()
                    raise Exception
                self.progress.show()
                class_model_label_shape = []
                class_model_label_shape.append(
                    [0, 0, 0, 0])  # The values is not important, but it needs to represent the shape of the labels.
                newBatch_feat = import_csv_to_predict_classification(class_pred_file)
                class_predict_tuplet = newBatch_feat, class_model_label_shape, class_model_name

                reset_tf()
                result_string = predict_classification(class_predict_tuplet)

                # Hide progressbar, show result of training and enable buttons
                self.progress.hide()
                self.output_field.setText(result_string)
                self.buttons_On()
            elif predict_model_msgBox.clickedButton() == predict_indent_btn:

                indent_model_name = QtGui.QFileDialog.getOpenFileName(self, 'Open CKPT File', '',
                                                                  '(*.meta)(*.index)(*.data-00000-of-00001)')
                if indent_model_name is "":
                    self.buttons_On()
                    raise Exception
                elif indent_model_name.endswith('.meta'):
                    indent_model_name = indent_model_name[:-5]
                elif indent_model_name.endswith('.index'):
                    indent_model_name = indent_model_name[:-6]
                elif indent_model_name.endswith('.data-00000-of-00001'):
                    indent_model_name = indent_model_name[:-20]
                class_pred_file = QtGui.QFileDialog.getOpenFileName(self, 'Open CSV File', '', '(*.csv)')
                if class_pred_file is "":
                    self.buttons_On()
                    raise Exception

                self.progress.show()
                indent_predict_tuplet = []
                indent_model_label_shape = []
                indent_model_label_shape.append(
                    [0])  # The values is not important, but it needs to represent the shape of the labels.

                newBatch_feat, string_status = import_csv_to_predict_classification(class_pred_file)

                indent_predict_tuplet = newBatch_feat, indent_model_label_shape, \
                                        indent_model_name
                reset_tf()
                result_string = predict_indent(indent_predict_tuplet)

                self.progress.hide()
                self.output_field.setText(result_string)
                self.buttons_On()

            elif predict_model_msgBox.clickedButton() == predict_cancel_model_btn:
                self.buttons_On()
                self.output_field.setEnabled("Welcome to this application\n\nCancelled")

        except:
            self.buttons_On()
            self.output_field.setText("Welcome to this application\n\nCancelled")
            self.output_field.append("\nDid you remember to select a file?")
            self.progress.hide()

    def convert_clicked(self):
        # Using try in case user types in unknown file or closes without choosing a file.
        try:
            self.buttons_Off()
            self.output_field.setText("CONVERT .TXT TO .CSV")
            self.output_field.append("\nChoose .txt-file to convert.")
            name_input_txt = QtGui.QFileDialog.getOpenFileName(self, 'Open text file', ' ', '*.txt')

            # Checking if there is a file
            if name_input_txt is "":
                raise Exception
            print(name_input_txt)
            self.output_field.append("\nName the .csv-file and where to save it.")
            name_output_csv = QtGui.QFileDialog.getSaveFileName(self, 'Save File', '', '*.csv')

            # Checking if there is a file
            if name_output_csv is "":
                raise Exception
            self.output_field.append("\n\nStarting convertion...\n\nThis may take a while, given the size of the .txt-file.")
            self.progress.show()

            # Number of features can be adjusted, but 500 is the optimized size for our program
            number_of_features = 500
            show_result = convert_txt_to_csv(number_of_features, name_input_txt, name_output_csv)
            self.output_field.setText(show_result)
            self.progress.hide()
            self.buttons_On()
        except:
            self.progress.hide()
            self.buttons_On()
            self.output_field.setText("Welcome to this Application\n\nError, did you remember to choose a file? Try again")

    def close_application(self):
        Exitprogram = QtGui.QMessageBox.question(self, 'Exit',
                                            "Are you sure you want to quit?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if Exitprogram == QtGui.QMessageBox.Yes:
            sys.exit()
        else:
            pass

# Running the GUI
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())

run()