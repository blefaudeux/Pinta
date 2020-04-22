#!/usr/bin/env python3

import settings
from data_processing.plot import polar_plot
from synthetic import polar
from train.engine_cnn import Conv

"""
Load a given engine, generate a couple of synthetic plots from it
"""


# Load the saved pytorch nn
training_settings = settings.get_defaults()

BATCH_SIZE = training_settings["batch_size"]
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]), training_settings["seq_length"]]

engine = Conv(
    logdir="logs/" + settings.get_name(),
    log_channel="NaiveConv",
    input_size=INPUT_SIZE,
    hidden_size=training_settings["hidden_size"],
    filename="trained/" + settings.get_name() + ".pt",
)

if not engine.valid:
    print("Failed loading the model, cannot continue")
    exit(-1)

# Generate data all along the curve
polar_data = polar.generate(engine, [5, 25], 5, 0.1, training_settings["seq_length"])

# Plot all that stuff
polar_plot(polar_data)
