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
from pinta.settings import Settings
import torch

LOG = logging.getLogger("DataLoad")


def _angle_split(data):
    data["twa_x"] = np.cos(np.radians(data["twa"]))
    data["twa_y"] = np.sin(np.radians(data["twa"]))
    return data


def load(filepath: Path, zero_mean_helm: bool = False) -> DataFrame:
    LOG.info("Loading %s" % filepath)
    data_frame = {
        ".json": lambda x: load_json(x, skip_zeros=True),
        ".pkl": pd.read_pickle,
    }[filepath.suffix](filepath)

    # Fix a possible offset in the rudder angle sensor
    if zero_mean_helm:
        data_frame["helm"] -= data_frame["helm"].mean()

    return data_frame


def load_folder(
    folder_path: Path,
    zero_mean_helm: bool,
    parallel_load: bool = True,
    max_number_sequences: int = -1,
) -> List[DataFrame]:
    # Get the matching files
    def valid(filepath):
        return os.path.isfile(filepath) and os.path.splitext(filepath)[1] in [
            ".json",
            ".pkl",
        ]

    filelist = [
        Path(os.path.join(folder_path, f))
        for f in os.listdir(folder_path)
        if valid(os.path.join(folder_path, f))
    ]

    if max_number_sequences > 0:
        filelist = filelist[:max_number_sequences]

    # Batch load all the files, saturate IO
    if parallel_load:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        results: List[DataFrame] = []
        barrier = pool.starmap_async(
            load,
            zip(filelist, repeat(zero_mean_helm)),
            callback=lambda x: results.append(x),
        )
        barrier.wait()
        assert len(results) > 0, "Could not load any data"
        return results[0]
    else:
        return [load(f, zero_mean_helm=zero_mean_helm) for f in filelist]


def to_training_set(raw_data: pd.DataFrame, settings: Settings):
    # Optionally replace the string settings by numerical tokens
    for cat_tokenize in settings.tokens:
        for k, v in settings.tokens[cat_tokenize].items():
            raw_data[cat_tokenize] = raw_data[cat_tokenize].replace(k, v)

    fused_inputs = settings.inputs + settings.tuning_inputs
    return TrainingSet.from_numpy(
        np.array(
            [raw_data[cat].values for cat in fused_inputs], dtype=float
        ).transpose(),
        np.array(
            [raw_data[cat].values for cat in settings.outputs], dtype=float
        ).transpose(),
    )


def split(
    raw_data: DataFrame, settings: Dict[str, Any]
) -> Tuple[TrainingSet, TrainingSet]:
    """
    Given a dataframe, split in between train and validation, requested inputs and outputs, and offset the samples
    if time prediction is involved
    """

    cat_in = settings["inputs"] + settings["tuning_inputs"]
    cat_out = settings["outputs"]
    ratio = settings["training_ratio"]
    train_size = int(len(raw_data) * ratio)
    LOG.info(
        "Training set is {} samples long. Selecting inputs: {} and outputs {}".format(
            train_size, cat_in, cat_out
        )
    )

    train, test = raw_data.iloc[:train_size], raw_data.iloc[train_size : len(raw_data)]

    train_inputs = np.array([train[cat].values for cat in cat_in], dtype=np.float32)
    test_inputs = np.array([test[cat].values for cat in cat_in], dtype=np.float32)

    # Move samples to first dimension, makes more sense if output is 1d
    train_output = np.array(
        [train[cat].values for cat in cat_out], dtype=np.float32
    ).transpose()
    test_output = np.array(
        [test[cat].values for cat in cat_out], dtype=np.float32
    ).transpose()

    return (
        TrainingSet(torch.tensor(train_inputs), torch.tensor(train_output)),
        TrainingSet(torch.tensor(test_inputs), torch.tensor(test_output)),
    )


def load_sets(raw_list, settings) -> List[TrainingSet]:
    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    loaded = []
    for x in raw_list:
        try:
            loaded.append(to_training_set(_angle_split(x), settings))
        except KeyError as e:
            LOG.warning(f"Could not load set. Cols are {x.columns}. Error is {e}")

    return loaded


def pack_sets(training_sets: List[TrainingSet]) -> TrainingSet:
    final_set = training_sets[0]

    for t_set in training_sets[1:]:
        final_set.append(t_set.inputs, t_set.outputs)

    return final_set
