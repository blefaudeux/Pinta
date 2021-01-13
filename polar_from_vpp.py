#!/usr/bin/env python3

import argparse
from pathlib import Path

import pinta.settings

from pinta.data_processing.load import load_folder, load_sets
from pinta.data_processing.plot import polar_plot
from pinta.data_processing.training_set import TrainingSetBundle
from pinta.model.model_factory import model_factory
from pinta.synthetic import polar


def run(args):
    """
    Load a given engine, generate a couple of synthetic plots from it
    """
    # Load the saved pytorch nn
    training_settings = settings.load(args.settings_path)

    # a bit hacky: get the normalization factors on the fly
    data = load_folder(Path(args.data_path), zero_mean_helm=False)
    datasplits = load_sets(data, training_settings)
    mean, std = TrainingSetBundle(datasplits).get_norm()

    model = model_factory(training_settings, model_path=args.model_path)
    model = model.to(device=settings.device)

    if not model.valid:
        print("Failed loading the model, cannot continue")
        exit(-1)

    # Generate data all along the curve
    polar_data = polar.generate(
        engine=model,
        wind_range=[5, 25],
        wind_step=5,
        angular_step=0.1,
        seq_len=training_settings["seq_length"],
        mean=mean,
        std=std,
        inputs=training_settings["inputs"],
    )

    # Plot all that stuff
    polar_plot(polar_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Polar plot from a trained model")

    parser.add_argument(
        "--data_path",
        action="store",
        help="path to the reference data, to get reasonable input estimates",
        default="data",
    )

    parser.add_argument(
        "--model_path", action="store", help="path to the .pt serialized model", default=None, required=True
    )

    parser.add_argument(
        "--settings_path", action="store", help="path to the json settings for the run", default=None, required=True
    )

    args = parser.parse_args()
    run(args)
