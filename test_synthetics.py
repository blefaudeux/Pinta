#!/usr/bin/env python3

from train.engine_cnn import Conv
from settings import INPUT_SIZE, HIDDEN_SIZE, NN_FILENAME
from synthetic import polar

"""
Load a given engine, generate a couple of synthetic plots from it
"""


# Load the saved pytorch nn
engine = Conv(logdir='logs/conv',
              input_size=INPUT_SIZE,
              hidden_size=HIDDEN_SIZE,
              filename=NN_FILENAME)

# Generate data all along the curve
polar_data = polar.generate(engine, [5, 25], 5, .1)

# Plot all that
# TODO: Ben
