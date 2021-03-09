import logging
import multiprocessing
import os
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pandas import DataFrame
from pinta.data_processing.training_set import TrainingSet
from pinta.data_processing.utils import load_json
import pandas as pd

LOG = logging.getLogger("DataLoad")


def _angle_split(data):
    data["twa_x"] = np.cos(np.radians(data["twa"]))
    data["twa_y"] = np.sin(np.radians(data["twa"]))
    return data


def load(filepath: Path, zero_mean_helm: bool = False) -> DataFrame:
    LOG.info("Loading %s" % filepath)
    data_frame = {".json": lambda x: load_json(x, skip_zeros=True), ".pkl": pd.read_pickle}[filepath.suffix](filepath)

    # Fix a possible offset in the rudder angle sensor
    if zero_mean_helm:
        data_frame["helm"] -= data_frame["helm"].mean()

    return data_frame


def load_folder(
    folder_path: Path, zero_mean_helm: bool, parallel_load: bool = True, max_number_sequences: int = -1
) -> List[DataFrame]:
    # Get the matching files
    def valid(filepath):
        return os.path.isfile(filepath) and os.path.splitext(filepath)[1] in [".json", ".pkl"]

    filelist = [
        Path(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if valid(os.path.join(folder_path, f))
    ]

    if max_number_sequences > 0:
        filelist = filelist[:max_number_sequences]

    # Batch load all the files, saturate IO
    if parallel_load:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        results: List[DataFrame] = []
        barrier = pool.starmap_async(load, zip(filelist, repeat(zero_mean_helm)), callback=lambda x: results.append(x))
        barrier.wait()
        return results[0]
    else:
        return [load(f, zero_mean_helm=zero_mean_helm) for f in filelist]


def to_training_set(raw_data, settings):
    return TrainingSet.from_numpy(
        np.array([raw_data[cat].values for cat in settings["inputs"]], dtype=np.float).transpose(),
        np.array([raw_data[cat].values for cat in settings["inputs_tuning"]], dtype=np.float).transpose(),
        np.array([raw_data[cat].values for cat in settings["outputs"]], dtype=np.float).transpose(),
    )


def split(raw_data: DataFrame, settings: Dict[str, Any]) -> Tuple[TrainingSet, TrainingSet]:
    """
    Given a dataframe, split in between train and validation, requested inputs and outputs, and offset the samples
    if time prediction is involved
    """

    cat_in = settings["inputs"]
    cat_out = settings["outputs"]
    ratio = settings["training_ratio"]
    train_size = int(len(raw_data) * ratio)
    LOG.info("Training set is {} samples long. Selecting inputs: {} and outputs {}".format(train_size, cat_in, cat_out))

    train, test = raw_data.iloc[:train_size], raw_data.iloc[train_size : len(raw_data)]

    train_inputs = np.array([train[cat].values for cat in cat_in], dtype=np.float)
    test_inputs = np.array([test[cat].values for cat in cat_in], dtype=np.float)

    # Move samples to first dimension, makes more sense if output is 1d
    train_output = np.array([train[cat].values for cat in cat_out], dtype=np.float).transpose()
    test_output = np.array([test[cat].values for cat in cat_out], dtype=np.float).transpose()

    return (
        TrainingSet(train_inputs, train_output),
        TrainingSet(test_inputs, test_output),
    )


def load_sets(raw_list, settings) -> List[TrainingSet]:
    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    return [to_training_set(_angle_split(x), settings) for x in raw_list]


def pack_sets(training_sets: List[TrainingSet]) -> TrainingSet:
    final_set = training_sets[0]

    for t_set in training_sets[1:]:
        final_set.append(t_set.inputs, t_set.outputs)

    return final_set
