from collections import Counter
import math
import pandas as pd
import numpy as np
import csv
import os
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def knn_model():

    # read the following values from User_Input file. Ideally these values should be read from UI / screen.
    training_data = ''
    test_data = ''
    k_str = '' 
    # print('training_data - BEFORE: ', training_data)
    # print('test_data - BEFORE: ', test_data)
    # print('K - BEFORE: ', k_str)
    with open('User_Input.txt', 'r') as userInputFile:
        header = userInputFile.readline()
        line = userInputFile.readline()
        if line != '':
            column = line.split(',')
            training_data = column[0]
            test_data = column[1]
            k_str = column[2]
        userInputFile.close()
    # print('training_data - AFTER: ', training_data)
    # print('test_data - AFTER: ', test_data)
    # print('K - AFTER: ', k_str)

    # based on uder's input, select the appropriate files for training data.
    raw_training_data_file = ''
    mod_training_data_file = ''
    # print('raw_training_data_file - BEFORE: ', raw_training_data_file)
    # print('mod_training_data_file - BEFORE: ', mod_training_data_file)
    with open('KNN_Config.txt', 'r') as knnConfigFile:
        line = knnConfigFile.readline()
        # print('line from KNNConfig file: ', line)
        while line != '':
            column = line.split(',')
            # print('column[0]: ', column[0])
            # print('training_data: ', training_data)
            if column[0].__eq__(training_data):
                # print('match found')
                raw_training_data_file = column[1]
                mod_training_data_file = column[2]
                break
            line = knnConfigFile.readline()
        knnConfigFile.close()
    # print('raw_training_data_file - AFTER: ', raw_training_data_file)
    # print('mod_training_data_file - AFTER: ', mod_training_data_file)


    # check if the mod_training_data_file exist. If exists, delete it. Need ato crete an empty file always.
    if os.path.isfile(mod_training_data_file):
        os.remove(mod_training_data_file)

    # based on user's input, select the training data files from KNN_Config file.

    # -------------------    Training data prepration - START ------------
    # read the data from file which contains:   column 1: participant id
    #                                           column 2: activiti id
    #                                           column 3: reading (e.g Acc_X)  
    # prepare the training data in the format: column 1: reading (e.g. Acc_X)
    #                                          column 2:  activity id
    with open(mod_training_data_file, 'w') as file1:

        with open(raw_training_data_file, 'r') as file2:

            org_line = file2.readline()
            # print('org_line: ', org_line)

            while org_line != '':
                mod_line = org_line.replace('\n','')
                column = mod_line.split(',')
                participant_id = column[0]
                activity_id = column[1]
                reading = column[2]

                file1.write( reading + '\t' + activity_id )
                file1.write('\n')

                org_line = file2.readline()

            file2.close()

        file1.close()

    # build 2D array from training data.
    train_data = np.loadtxt(mod_training_data_file)
    # print('train_data: ', train_data)

    # -------------------    Training data prepration - END ------------

    # read the input data from UI / screen
    # input_point = [0.3307439]
    test_data_float = float(test_data)
    input_point = [test_data_float]
    # print('input_point', input_point)

    # calculate the distance between input & all other points in the training data.
    # and store the distance in an array.
    distance_array = []
    # print('distance_array - START: ', distance_array)
    for i, point_in_train_data in enumerate(train_data):

        # print('------------ i --------------:  ', i)
        # calculate the distance netween 2 points
        sum_squared_distance = 0
        for i in range(len(point_in_train_data[:-1])):
            sum_squared_distance += math.pow(point_in_train_data[:-1][i] - input_point[i], 2)

        distance = math.sqrt(sum_squared_distance)
        # print('distance: ', distance)

        # store the distance & index into an array.
        distance_array.append((distance, i))
        # print('distance_array: ', distance_array)

    # print('distance_array - END: ', distance_array)

    # Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_distance_array = sorted(distance_array)
    # print('sorted_distance_array; ', sorted_distance_array)

    # read the K value from UI
    k = int(k_str)
    # print('k: ', k)

    # Pick the first K entries from the sorted collection
    k_nearest_distances = sorted_distance_array[:k]
    # print('k_nearest_distances: ', k_nearest_distances)

    # Get the labels of the selected K entries
    k_nearest_labels = [train_data[j][-1] for distance, j in k_nearest_distances]
    # print('k_nearest_labels: ', k_nearest_labels)

    # since the Mode represents the most frequently repeated value in a data set, trea Mode as the class that the input belongs to.
    output = Counter(k_nearest_labels).most_common(1)[0][0]    
    # print('output: ', output)


    # ----------- New Implementation of KNN - START --------------------

    total_data = train_data 
    print('total_data: ', total_data)

    # The 2D array total_data contains 2 column; Acclerometer / Gyroscope Reading, Acivity ID
    # X - axis is independent variable. i.e.  Acclerometer / Gyroscope Reading
    # Y - axis is dependent vatiable. i.e. Activity ID
    # Load the 1st column to X-axis and 2nd column to Y-axis.
    # The X-axis should be a 2D array and Y-axis should be a 1D array.

    # build a 2D array for X-axis data.
    new_training_data_file = 'New_Acc_X.csv'
    with open(new_training_data_file, 'w') as file1:

        with open(mod_training_data_file, 'r') as file2:

            org_line = file2.readline()
            # print('org_line: ', org_line)

            while org_line != '':
                mod_line = org_line.replace('\n','')
                column = mod_line.split('\t')
                reading = column[0]
                activity_id = column[1]

                file1.write(reading)
                file1.write('\n')

                org_line = file2.readline()

            file2.close()

        file1.close()

    # build 2D array from training data.
    X_total = np.loadtxt(new_training_data_file)
    print('X_total: ', X_total)

    X_total_1 = train_data
    Y_total = []
    # print('X_total - BEFORE: ', Y_total)

    for i in range(len(total_data)):

        X_total_1[i][0] = total_data[i][0]
        Y_total.append(total_data[i][1])

    print('X_total_1 - AFTER: ', X_total_1)
    # print('X_total - AFTER: ', Y_total)

    # Split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size= 0.25, random_state=38, stratify = Y_total)
    print('X_train: ', X_train)
    print('X_test: ', X_test)
    print('Y_train: ', Y_train)
    print('Y_test: ', Y_test)

    

    # Fitting the KNN model
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(X_train, Y_train)

    # Prediction of test set
    prediction_knn = knn.predict(X_test)
    print('prediction_knn: ', prediction_knn)

    # build the confusion matrix.
    # matrix = confusion_matrix()
    matrix = confusion_matrix(Y_test, prediction_knn)
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # ----------- New Implementation of KNN - END --------------------


knn_model()
