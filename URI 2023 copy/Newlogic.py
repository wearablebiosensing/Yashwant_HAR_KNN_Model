import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import tsfel
import json

# Sample data for demonstration
# Replace this with the actual data from X_train.txt
X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
subject_train = np.loadtxt('subject_train.txt')

# Create separate arrays for each activity
WALKING = []
WALKING_UPSTAIRS = []
WALKING_DOWNSTAIRS = []
SITTING = []
STANDING = []
LAYING = []

# Function to append the selected axis values to the corresponding activity array
def append_to_activity():
    # Get the selected axis from the dropdown
    axis_wanted = axis_var.get()

    # Axis names mapping
    axis_names = {
        0: "Accelerometer X",
        1: "Accelerometer Y",
        2: "Accelerometer Z",
        3: "Gyroscope X",
        4: "Gyroscope Y",
        5: "Gyroscope Z"
    }

    # Iterate through X_train and append the selected axis value to the corresponding activity array
    for i in range(len(X_train)): 
        activity_id = int(Y_train[i])
        if activity_id == 1:
            WALKING.append(X_train[i][axis_wanted])
        elif activity_id == 2:
            WALKING_UPSTAIRS.append(X_train[i][axis_wanted])
        elif activity_id == 3:
            WALKING_DOWNSTAIRS.append(X_train[i][axis_wanted])
        elif activity_id == 4:
            SITTING.append(X_train[i][axis_wanted])
        elif activity_id == 5:
            STANDING.append(X_train[i][axis_wanted])
        elif activity_id == 6:
            LAYING.append(X_train[i][axis_wanted])


    print("---------------------------------WALKING--------------------------------------",WALKING)
    print("---------------------------------WALKING_UPSTAIRS--------------------------------------",WALKING_UPSTAIRS)
    print("---------------------------------WALKING_DOWNSTAIRS--------------------------------------",WALKING_DOWNSTAIRS)
    print("---------------------------------SITTING--------------------------------------",SITTING)
    print("---------------------------------STANDING--------------------------------------",STANDING)
    print("---------------------------------LAYING--------------------------------------",LAYING)
    
    
    cfg_file = tsfel.get_features_by_domain("statistical")
    walking_features = tsfel.time_series_features_extractor(cfg_file, np.array(WALKING))
    walking_upstairs_features = tsfel.time_series_features_extractor(cfg_file, np.array(WALKING_UPSTAIRS))
    walking_downstairs_features = tsfel.time_series_features_extractor(cfg_file, np.array(WALKING_DOWNSTAIRS))
    sitting_features = tsfel.time_series_features_extractor(cfg_file, np.array(SITTING))
    standing_features = tsfel.time_series_features_extractor(cfg_file, np.array(STANDING))
    laying_features = tsfel.time_series_features_extractor(cfg_file, np.array(LAYING))

    # Print the calculated features (you can also save them to a file if needed)
    print("Walking Features:")
    print(walking_features)
    print("Walking Upstairs Features:")
    print(walking_upstairs_features)
    print("Walking Downstairs Features:")
    print(walking_downstairs_features)
    print("Sitting Features:")
    print(sitting_features)
    print("Standing Features:")
    print(standing_features)
    print("Laying Features:")
    print(laying_features)

    walking_features_dict = walking_features.to_dict(orient='list')
    walking_upstairs_features_dict = walking_upstairs_features.to_dict(orient='list')
    walking_downstairs_features_dict = walking_downstairs_features.to_dict(orient='list')
    sitting_features_dict = sitting_features.to_dict(orient='list')
    standing_features_dict = standing_features.to_dict(orient='list')
    laying_features_dict = laying_features.to_dict(orient='list')

    with open("extracted_features.txt", "w") as file:
        file.write("Walking Features:\n")
        file.write(str(walking_features_dict) + "\n")
        file.write("Walking Upstairs Features:\n")
        file.write(str(walking_upstairs_features_dict) + "\n")
        file.write("Walking Downstairs Features:\n")
        file.write(str(walking_downstairs_features_dict) + "\n")
        file.write("Sitting Features:\n")
        file.write(str(sitting_features_dict) + "\n")
        file.write("Standing Features:\n")
        file.write(str(standing_features_dict) + "\n")
        file.write("Laying Features:\n")
        file.write(str(laying_features_dict) + "\n")    

    
    # Create a boxplot for each activity
    plt.figure()
    plt.boxplot([WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING],
                labels=["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"])
    plt.title(f"Boxplot for {axis_names[axis_wanted]} for Different Activities")
    plt.xlabel("Activity")
    plt.ylabel(f"{axis_names[axis_wanted]} Value")
    plt.show()

# Create the GUI
root = tk.Tk()
root.title("Feature Analysis")
root.geometry("400x250")
# Dropdown for selecting the axis
axis_var = tk.IntVar()
axis_names = {
    0: "Accelerometer X",
    1: "Accelerometer Y",
    2: "Accelerometer Z",
    3: "Gyroscope X",
    4: "Gyroscope Y",
    5: "Gyroscope Z"
}
axis_dropdown = ttk.Combobox(root, textvariable=axis_var, values=list(axis_names.keys()))
axis_dropdown.pack()


# Button to trigger the append operation and graph the box plot
append_button = tk.Button(root, text="Append to Activity and Graph Boxplot", command=append_to_activity)
append_button.pack()

# Start the main loop
root.mainloop()
