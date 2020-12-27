#!/usr/bin/env python3

import argparse
from pathlib import Path

import settings
from data_processing.load import load_folder, load_sets
from data_processing.plot import polar_plot
from data_processing.training_set import TrainingSetBundle
from model.model_factory import model_factory
from synthetic import polar


def run(args):
    """
    Load a given engine, generate a couple of synthetic plots from it
    """
    # Load the saved pytorch nn
    training_settings = settings.get_default_params()
    # a bit hacky: get the normalization factors on the fly
    data = load_folder(Path("data"), zero_mean_helm=False)
    datasplits = load_sets(data, training_settings)
    mean, std = TrainingSetBundle(datasplits).get_norm()

    if args.model_path:
        model_path = args.model_path
    else:
        model_path = "trained/" + settings.get_name() + ".pt"

    engine = model_factory(training_settings, model_path=model_path)
    engine = engine.to(device=settings.device)

    if not engine.valid:
        print("Failed loading the model, cannot continue")
        exit(-1)

    # Generate data all along the curve
    polar_data = polar.generate(engine, [5, 25], 5, 0.1, training_settings["seq_length"], mean, std)

    # Plot all that stuff
    polar_plot(polar_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate Polar plot from a trained model")

    parser.add_argument(
        "--model_path",
        action="store",
        help="path to the .pt serialized model",
    )

    args = parser.parse_args()
    run(args)
