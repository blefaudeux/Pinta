import os

from data_processing.nmea2pandas import load_json
from data_processing.whitening import whiten_angle
from data_processing.split import split
from settings import TrainingSet
import numpy as np


def _angle_split(data):
    data['wind_angle_x'] = np.cos(np.radians(data['wind_angle']))
    data['wind_angle_y'] = np.sin(np.radians(data['wind_angle']))
    return data


def load_folder(folder_path, clean_data=True, whiten_data=True):
    def valid(filepath):
        return os.path.isfile(filepath) and os.path.splitext(filepath)[1] == ".json"

    filelist = [os.path.join(folder_path, f) for f in os.listdir(
        folder_path) if valid(os.path.join(folder_path, f))]
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

    training_data = TrainingSet(input=[], output=[])
    testing_data = TrainingSet(input=[], output=[])

    for pair in raw:
        # Handle the angular coordinates discontinuity -> split x/y components
        raw_data = _angle_split(pair[0])
        raw_data_aug = _angle_split(pair[1])

        # Split in between training and test
        train, test = split(raw_data, settings)
        train_r, test_r = split(raw_data_aug, settings)

        # All the sub-datasets are not coherent over time.
        # Keep a list of them, do not concatenate straight
        training_data.input.append(train.input)
        training_data.input.append(train_r.input)

        training_data.output.append(train.output)
        training_data.output.append(train_r.output)

        testing_data.input.append(test.input)
        testing_data.input.append(test_r.input)

        testing_data.output.append(test.output)
        testing_data.output.append(test_r.output)

    return training_data, testing_data
