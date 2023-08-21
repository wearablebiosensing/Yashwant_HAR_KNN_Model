from collections import Counter
import math
import pandas as pd
import numpy as np
import csv


def knn_model():

    # based on user's input, select the training data files from KNN_Config file.

    # -------------------    Training data prepration - START ------------
    # read the data from file which contains:   column 1: participant id
    #                                           column 2: activiti id
    #                                           column 3: reading (e.g Acc_X)  
    # prepare the training data in the format: column 1: reading (e.g. Acc_X)
    #                                          column 2:  activity id
    with open('Acc_X_Activity.csv', 'w') as file1:

        with open('Participant_Acc_X.csv', 'r') as file2:

            org_line = file2.readline()

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
    train_data = np.loadtxt('Acc_X_Activity.csv')
    # print('train_data: ', train_data)

    # -------------------    Training data prepration - END ------------

    # read the input data from UI / screen
    input_point = [0.3307439]
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
    k = 3
    # print('k: ', k)

    # Pick the first K entries from the sorted collection
    k_nearest_distances = sorted_distance_array[:k]
    # print('k_nearest_distances: ', k_nearest_distances)

    # Get the labels of the selected K entries
    k_nearest_labels = [train_data[j][-1] for distance, j in k_nearest_distances]
    # print('k_nearest_labels: ', k_nearest_labels)

    # since the Mode represents the most frequently repeated value in a data set, trea Mode as the class that the input belongs to.
    output = Counter(k_nearest_labels).most_common(1)[0][0]    
    print('output: ', output)


knn_model()

