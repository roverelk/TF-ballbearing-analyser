# HENSIKT:  Importere egne tall inn til programmet.
#
# LOG
# DATE------INIT----WHAT--------COMMENT----------------------------------
# 04.01.17  ekk     Created     Kilde: https://www.tensorflow.org/versions/master/tutorials/mnist/pros/
# 04.01.17  ekk     Comment     Programmet er tregt på min PC. Programmet gir ut resultater undervei mens det kjører.
# 24.01.17  ekk     Comment     Fått ny innsikt og skal prøve å forandre det til et brukbart program.
# 25.01.17  ekk     Comment     Dette programmet virker BEST PÅ GPU!
# 30.01.17  ekk     Comment     gradient_length var ikke styrt herfra før nå, er rett opp i. må ikke røres fra
#                               csv_to_array_enCsv.py
# 31.01.17  ekk     Comment     Removed unnessecary imports.
# 08.01.17  ekk     Comment     The model is now working. It can save a model, reload a saved model, continue working
#                               on it and it can predict a result based on a input sample.
# 13.02.17  ekk     Remodel     Remodeled the names and the logistic structure of the whole program into more separated
#                               scripts as it will be neccesary to alter them for spessified tasks.

#COMPATIBLE WITH:
from BA__txt_to_csv__v5                         import txt_to_csv_def       as convert_txt_to_csv
from CA__modelClassifyProblem__csv_to_array__v5 import csv_to_array_def     as import_csv_for_classification
from CB__TF_trainModel__v9                      import array_to_TF_def      as train_model_classification
from DA__modelIndentSize__csv_to_array__v5      import csv_to_array_def     as import_csv_for_indent
from DB__TF_trainModel__v9                      import array_to_TF_def      as train_model_indent
from EA__prepForPredClass__csv_to_array__v5     import csv_to_array_def     as import_csv_to_predict_classification
from EB__predictClassification__v4              import TF_to_prediction_def as predict_classification
from FA__prepForPredIndent__csv_to_array__v5    import csv_to_array_def     as import_csv_to_predict_indent
from FB__predictIndent__v4                      import TF_to_prediction_def as predict_indent

#Might move this script to it's own file.
from CB__TF_trainModel__v9 import reset as reset_tf


#### MASTER CONTROL #############################################################
runConversion   = False     # Run convertion .txt -> .csv
trainTFClass    = False     # Run TF training for classification model
trainTFIndent   = True     # Run TF training for indent model
predTFClass     = False     # Use model to predict type of damage
predTFIndent    = False     # Use model to predict size of indent

# Variablene under er på vei vekk.
# .ckpt-fil
fil_navn = "model_fault_inner_v1.ckpt"
iteration_number = 0
#################################################################################


#### CONTROLPANEL: convert_txt_to_csv ###########################################
number_of_features = 500
name = "Innerring0-014"
name_input_txt     = name + ".txt"
name_output_csv    = name + "CSV_" + str(number_of_features) + ".csv"

#### KONTROLLPANEL: import_csv_for_classification ###############################
# Features
class_feature = []
class_feature.append("BaselineCSV_500.csv")
class_feature.append("YtterringCSV_500_0-007.csv")
class_feature.append("YtterringCSV_500_0-014.csv")
class_feature.append("YtterringCSV_500_0-021.csv")
class_feature.append("InnerringCSV_500_0-007.csv")
class_feature.append("InnerringCSV_500_0-014.csv")
class_feature.append("InnerringCSV_500_0-021.csv")
class_feature.append("BallCSV_500_0-007.csv")
class_feature.append("BallCSV_500_0-014.csv")
class_feature.append("BallCSV_500_0-021.csv")

# Labes
class_label = []
class_label.append([1, 0, 0, 0])  # No error
class_label.append([0, 1, 0, 0])  # Fault on outer ring
class_label.append([0, 1, 0, 0])
class_label.append([0, 1, 0, 0])
class_label.append([0, 0, 1, 0])  # Fault on inner ring
class_label.append([0, 0, 1, 0])
class_label.append([0, 0, 1, 0])
class_label.append([0, 0, 0, 1])  # Fault on at least one ball
class_label.append([0, 0, 0, 1])
class_label.append([0, 0, 0, 1])
# Each label corresponds to the feature-files in the same order.

class_folder = "database_csv"
#"folder" is where the featured .csv-files are stored. Same base folder.
class_percent = 0.85
#"percent" defines the

#### CONTROLPANEL: train_model_classification ###################################
class_gradient_length                 = 0.000001 #Change to change to the learningsteps
class_print_intermidiate_values_train = False
class_print_intermidiate_values_test  = False
class_print_error_rate                = True
class_print_graph                     = True
class_new_file                          = False #False = opprett ny fil
                                        #True  = last gammel fil med navn 'fil_navn'


#### KONTROLLPANEL: import_csv_for_indent #######################################
# Features
indent_feature = []
indent_feature.append("BaselineCSV_500.csv")
indent_feature.append("InnerringCSV_500_0-007.csv")
indent_feature.append("InnerringCSV_500_0-014.csv")
indent_feature.append("InnerringCSV_500_0-021.csv")

# Labels
indent_label = []
indent_label.append([0])
indent_label.append([7])
indent_label.append([14])
indent_label.append([21])

indent_folder = "database_csv"
indent_percent = 0.85

#### CONTROLPANEL: train_model_indent ###########################################
indent_gradient_length                 = 0.000001 #Change to change to the learningsteps
indent_print_intermidiate_values_train = False
indent_print_intermidiate_values_test  = False
indent_print_error_rate                = True
indent_print_graph                     = True
indent_new_file                          = False #False = opprett ny fil
                                        #True  = last gammel fil med navn 'fil_navn'

#### CONTROLPANEL: import_csv_to_predict_classification #########################
class_pred_folder = "database_csv"
class_pred_file = "YtterringCSV_500_0-007.csv"

#### CONTROLPANEL: predict_classification #######################################
class_model_name = "model_class_all.ckpt"
class_model_folder = "model_classification"
class_model_iteration = 2650
class_model_label_shape = []
class_model_label_shape.append([0, 0, 0, 0])  #The values is not important, but it needs to represent the shape of the labels.

#### CONTROLPANEL: import_csv_to_predict_indent #################################
indent_pred_folder = "database_csv"
indent_pred_file = "YtterringCSV_500_0-014.csv"

#### CONTROLPANEL: predict_indent ###############################################
# Outer ring mdel
indent_model_outer_name      = "model_fault_outer_v4.ckpt"
indent_model_outer_folder    = "model_indent_outer"
indent_model_outer_iteration = 1350
# Inner ring model
indent_model_inner_name      = "IN_PROGRESS.ckpt"
indent_model_inner_folder    = "model_indent_inner"
indent_model_inner_iteration = 1350
# Ball model
indent_model_ball_name       = "IN_PROGRESS"
indent_model_ball_folder     = "IN_PROGRESS.ckpt"
indent_model_ball_iteration  = 1350

indent_model_label_shape = []
indent_model_label_shape.append([0])  #The values is not important, but it needs to represent the shape of the labels.

#### Global variables
newSvar_feat = []
newBatch_lab = []
newBatch_feat = []
newSvar_lab = []
type_of_class = "-1"

if runConversion:
        print("--------------------------------------------")
        print("Run: convert_txt_to_csv")
        convert_txt_to_csv(number_of_features, name_input_txt, name_output_csv)
        print("\nSuccessfully ran: convert_txt_to_csv")

if trainTFClass:
    print("--------------------------------------------")
    print("Run: import_csv_for_classification")
    newBatch_feat, newBatch_lab, newSvar_feat, newSvar_lab = import_csv_for_classification(class_feature, class_label, class_percent)
    print("\nSuccessfully ran: import_csv_for_classification")
    print("--------------------------------------------")
    print("Run: train_model_classification\n")
    class_train_tuplet = newBatch_feat, newBatch_lab, newSvar_feat, newSvar_lab, class_gradient_length, class_print_intermidiate_values_train, \
            class_print_intermidiate_values_test, class_print_error_rate, class_print_graph, fil_navn, iteration_number, class_new_file
    reset_tf()
    train_model_classification(class_train_tuplet)
    print("\nSuccessfully ran: train_model_classification")

if trainTFIndent:
    print("--------------------------------------------")
    print("Run: import_csv_for_indent\n")
    newBatch_feat, newBatch_lab, newSvar_feat, newSvar_lab = import_csv_for_indent(indent_folder, indent_feature, indent_label, indent_percent)
    print("\nSuccessfully ran: import_csv_for_indent")
    print("--------------------------------------------")
    print("Run: train_model_indent\n")
    indent_train_tuplet = newBatch_feat, newBatch_lab, newSvar_feat, newSvar_lab, indent_gradient_length, indent_print_intermidiate_values_train, \
            indent_print_intermidiate_values_test, indent_print_error_rate, indent_print_graph, fil_navn, iteration_number, indent_new_file
    reset_tf()
    train_model_indent(indent_train_tuplet)
    print("\nSuccssfully ran: train_model_indent")

if predTFClass:
    print("--------------------------------------------")
    print("Run: import_csv_for_classification\n")
    newBatch_feat = import_csv_to_predict_classification(class_pred_folder, class_pred_file)
    print("\nSuccessfully ran: import_csv_for_classification")
    print("--------------------------------------------")
    print("Run: predict_classification\n")
    class_predict_tuplet = newBatch_feat, class_model_label_shape, class_model_folder, class_model_name, \
                           class_model_iteration
    reset_tf()
    type_of_class = predict_classification(class_predict_tuplet)
    print("\nSuccessfully ran: predict_classification")

if predTFIndent:
    if not predTFClass:
        print("--------------------------------------------")
        print("Run: import_csv_for_indent\n")
        newBatch_feat = import_csv_to_predict_indent(indent_pred_folder, indent_pred_file)
        print("\nSuccessfully ran: import_csv_for_indent")
    print("--------------------------------------------")
    print("Run: predict_indent\n")
    reset_tf()
    indent_predict_tuplet = []
    if type_of_class == 1:  # outer ring
        print("PREDICTION FOR OUTER RING INDENT")
        indent_predict_tuplet = newBatch_feat, indent_model_label_shape, indent_model_outer_folder, \
                                indent_model_outer_name, indent_model_outer_iteration
    if type_of_class == 2:  # inner ring
        print("PREDICTION FOR INNWE RING INDENT")
        indent_predict_tuplet = newBatch_feat, indent_model_label_shape, indent_model_inner_folder, \
                                indent_model_inner_name, indent_model_inner_iteration
    if type_of_class == 3:  # ball
        print("PREDICTION FOR BALL INDENT")
        indent_predict_tuplet = newBatch_feat, indent_model_label_shape, indent_model_ball_folder, \
                                indent_model_ball_name, indent_model_ball_iteration
    predict_indent(indent_predict_tuplet)
    print("\nSuccessfully ran: predict_indent")

print("--------------------------------------------")
