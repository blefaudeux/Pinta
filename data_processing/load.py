import os

from data_processing.nmea2pandas import load_json
from data_processing.whitening import whiten_angle
from data_processing.split import split
from settings import Dataframe
import numpy as np


def _angle_split(data):
    data['wind_angle_x'] = np.cos(np.radians(data['wind_angle']))
    data['wind_angle_y'] = np.sin(np.radians(data['wind_angle']))
    return data


def load_folder(folder_path, clean_data=True, whiten_data=True):
    def valid(filepath):
        return os.path.isfile(filepath) and os.path.splitext(filepath)[1] == "json"

    filelist = [f for f in os.listdir(folder_path) if valid(f)]
    return [load(f, clean_data, whiten_data) for f in filelist]


def load(filename, clean_data=True, whiten_data=True):
    print(f"Loading {filename}")
    data_frame = load_json(filename, skip_zeros=True)

    # Fix a possible offset in the rudder angle sensor
    if clean_data:
        data_frame['rudder_angle'] -= data_frame['rudder_angle'].mean()

    # Whiten the data, in that the boat supposedely goes at the same speed port and starboard
    if whiten_data:
        df_white_angle = whiten_angle(data_frame)
    else:
        df_white_angle = None

    return [data_frame, df_white_angle]


def package_data(raw, settings):
    if not isinstance(raw, list):
        raw = [raw]

    training_data = Dataframe(input=[], output=[])
    testing_data = Dataframe(input=[], output=[])

    for pair in raw:
        # Handle the angular coordinates discontinuity -> split x/y components
        raw_data = _angle_split(pair[0])
        raw_data_aug = _angle_split(pair[1])

        # Small debug plot, have a look at the data
        # data_plot = settings["inputs"] + settings["outputs"]
        # plt.parrallel_plot([raw_data[i]
        #                     for i in data_plot], data_plot, "Dataset plot")

        # Split in between training and test
        train_in, train_out, test_in, test_out = split(
            raw_data, settings)
        train_in_r, train_out_r, test_in_r, test_out_r = split(
            raw_data_aug, settings)

        training_data.input += train_in_r
        training_data.input += train_in

        training_data.output += train_out
        training_data.output += train_out_r

        testing_data.input += test_in
        testing_data.input += test_in_r

        testing_data.output += test_out
        testing_data.output += test_out_r

    return training_data, testing_data
