import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('X_train.txt')
Y_train = np.loadtxt('Y_train.txt')
subject_train = np.loadtxt('subject_train.txt')

window = tk.Tk()
window.title("Feature Analysis")
window.geometry("400x200")
feature_options = [
    "Max",
    "Min",
    "Range",
    "Std",
    "Mean",
    "Median",
    "Mode",
    "Root Mean Square",
    "Skewness",
    "Variance",
]
Data_Options = ["Accelerometer", "Gyroscope"]
feature_label = tk.Label(window, text="Select Feature:")
feature_label.pack()
feature_var = tk.StringVar(window)
feature_dropdown = ttk.Combobox(
    window, textvariable=feature_var, values=feature_options
)
feature_dropdown.pack()
data_type_label = tk.Label(window, text="Select Data Type:")
data_type_label.pack()
data_type_var = tk.StringVar(window)
data_type_dropdown = ttk.Combobox(
    window, textvariable=data_type_var, values=Data_Options
)
data_type_dropdown.pack()
def generate_box_plots():
    selected_feature = feature_var.get()
    selected_data_type = data_type_var.get()
    if selected_data_type == "Accelerometer":
        data_columns = [0, 1, 2]
    else:
        data_columns = [3, 4, 5]
    activity_labels = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING",
    ]
    feature_values = []
    for activity_id in range(1, 7):
        activity_indices = np.where(Y_train == activity_id)[0]
        activity_data = X_train[activity_indices][:, data_columns]
        if selected_feature == "Max":
            feature_values.append(np.max(activity_data, axis=0))
        elif selected_feature == "Min":
            feature_values.append(np.min(activity_data, axis=0))
        elif selected_feature == "Range":
            feature_values.append(np.ptp(activity_data, axis=0))
        elif selected_feature == "Std":
            feature_values.append(np.std(activity_data, axis=0))
        elif selected_feature == "Mean":
            feature_values.append(np.mean(activity_data, axis=0))
        elif selected_feature == "Root Mean Square":
            feature_values.append(np.sqrt(np.mean(activity_data ** 2, axis=0)))
        elif selected_feature == "Skewness":
            feature_values.append(pd.DataFrame(activity_data).skew().values)
        elif selected_feature == "Variance":
            feature_values.append(np.var(activity_data, axis=0))


#Graphing Code
    fig, ax = plt.subplots()
    ax.boxplot(feature_values)
    ax.set_xticklabels(activity_labels, rotation=45)
    ax.set_xlabel("Activity")
    ax.set_ylabel(selected_feature)
    ax.set_title(f"Box Plot of {selected_feature} ({selected_data_type} Data)")
    plt.show()
generate_button = tk.Button(
    window, text="Generate Box Plots", command=generate_box_plots
)
generate_button.pack()
window.mainloop()
