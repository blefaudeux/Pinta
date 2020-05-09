#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import settings
from data_processing.load import load_folder, load_sets
from data_processing.plot import polar_plot
from data_processing.training_set import TrainingSetBundle
from model.engine_cnn import Conv
from model.engine_dilated_conv import TemporalModel
from settings import ModelType
from synthetic import polar


def model_factory(params: Dict[str, Any], filename: str):

    if params["model_type"] == ModelType.Conv:
        INPUT_SIZE = [len(params["inputs"]), params["seq_length"]]

        dnn = Conv(
            logdir="logs/" + settings.get_name() + str(datetime.now()),
            input_size=INPUT_SIZE,
            hidden_size=params["hidden_size"],
            kernel_size=params["conv_width"],
            filename=filename,
            log_channel="DNN  ",
        )

    if params["model_type"] == ModelType.DilatedConv:
        dnn = TemporalModel(
            len(params["inputs"]),
            len(params["outputs"]),
            params["conv_width"],
            dropout=0.25,
            channels=1024,
            filename=filename,
        )

    dnn.to(settings.device)
    return dnn


def run(args):
    """
    Load a given engine, generate a couple of synthetic plots from it
    """
    # Load the saved pytorch nn
    training_settings = settings.get_default_params()
    # a bit hacky: get the normalization factors on the fly
    mean, std = TrainingSetBundle(
        load_sets(load_folder(Path("data")), training_settings)
    ).get_norm()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = "trained/" + settings.get_name() + ".pt"

    engine = model_factory(training_settings, filename=model_path)
    engine = engine.to(device=settings.device)

    if not engine.valid:
        print("Failed loading the model, cannot continue")
        exit(-1)

    # Generate data all along the curve
    polar_data = polar.generate(
        engine, [5, 25], 5, 0.1, training_settings["seq_length"], mean, std
    )

    # Plot all that stuff
    polar_plot(polar_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Polar plot from a trained model")

    parser.add_argument(
        "--model_path", action="store", help="path to the .pt serialized model",
    )

    args = parser.parse_args()
    run(args)
