import logging
import os
from pathlib import Path
from typing import List

import numpy as np

from data_processing.nmea2pandas import load_json
from data_processing.training_set import TrainingSet
from data_processing.whitening import whiten_angle

LOG = logging.getLogger("DataLoad")


def _angle_split(data):
    data["wind_angle_x"] = np.cos(np.radians(data["wind_angle"]))
    data["wind_angle_y"] = np.sin(np.radians(data["wind_angle"]))
    return data


def load_folder(folder_path: Path, clean_data=True, whiten_data=True):
    def valid(filepath):
        return os.path.isfile(filepath) and os.path.splitext(filepath)[1] == ".json"

    filelist = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if valid(os.path.join(folder_path, f))
    ]

    return [load(Path(f), clean_data, whiten_data) for f in filelist]


def load(filepath: Path, clean_data=True, whiten_data=True):
    LOG.info("Loading %s" % filepath)
    data_frame = load_json(filepath, skip_zeros=True)

    # Fix a possible offset in the rudder angle sensor
    if clean_data:
        data_frame["rudder_angle"] -= data_frame["rudder_angle"].mean()

    # Whiten the data, in that the boat supposedely goes at the same speed
    # port and starboard
    if whiten_data:
        df_white_angle = whiten_angle(data_frame)
    else:
        df_white_angle = None

    return [data_frame, df_white_angle]


def to_training_set(raw_data, settings):
    cat_in = settings["inputs"]
    cat_out = settings["outputs"]

    # Move samples to first dimension, makes more sense if output is 1d
    inputs = np.array([raw_data[cat].values for cat in cat_in]).transpose()
    outputs = np.array([raw_data[cat].values for cat in cat_out]).transpose()

    return TrainingSet.from_numpy(inputs, outputs)


def split(raw_data, settings):
    cat_in = settings["inputs"]
    cat_out = settings["outputs"]
    ratio = settings["training_ratio"]

    train_size = int(len(raw_data) * ratio)
    LOG.info("Training set is {} samples long".format(train_size))

    train, test = raw_data.iloc[:train_size], raw_data.iloc[train_size : len(raw_data)]

    train_inputs = np.array([train[cat].values for cat in cat_in])
    test_inputs = np.array([test[cat].values for cat in cat_in])

    # Move samples to first dimension, makes more sense if output is 1d
    train_output = np.array([train[cat].values for cat in cat_out]).transpose()
    test_output = np.array([test[cat].values for cat in cat_out]).transpose()

    return (
        TrainingSet(train_inputs, train_output),
        TrainingSet(test_inputs, test_output),
    )


def load_sets(raw, settings) -> List[TrainingSet]:
    if not isinstance(raw, list):
        raw = [raw]

    training_sets = []

    for pair in raw:
        # Handle the angular coordinates discontinuity -> split x/y components
        raw_data = _angle_split(pair[0])
        raw_data_aug = _angle_split(pair[1])

        # Save both sets, original and flipped
        training_sets.append(to_training_set(raw_data, settings))
        training_sets.append(to_training_set(raw_data_aug, settings))

    return training_sets


def pack_sets(training_sets: List[TrainingSet]) -> TrainingSet:
    final_set = training_sets[0]

    for t_set in training_sets[1:]:
        final_set.append(t_set.inputs, t_set.outputs)

    return final_set
