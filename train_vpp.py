#!/usr/bin/env python3

import numpy as np
from data_processing import plot as plt
from data_processing.load import load, load_folder, package_data
# from train.engine_rnn import ConvRNN
from train.engine_cnn import Conv
import settings


training_settings = settings.get_defaults()

# Load the dataset + some data augmentation
training_data, testing_data = package_data(
    load_folder('data'), training_settings)

# ConvRNN
BATCH_SIZE = training_settings["batch_size"]
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]),
              training_settings["seq_length"]]

print(f"Training on {len(training_data.input)} samples. Batch is {BATCH_SIZE}")
dnn = Conv(logdir='logs/conv',
           input_size=INPUT_SIZE,
           hidden_size=training_settings["hidden_size"],
           filename=training_settings["network_filename"])

if not dnn.valid():
    dnn.fit(training_data,
            testing_data,
            epoch=EPOCH,
            batch_size=BATCH_SIZE,
            seq_len=training_settings["seq_length"])
    dnn.save(training_settings["network_filename"])


testScore = dnn.evaluate(
    testing_data,
    seq_len=training_settings["seq_length"])

print('Final test Score: %.2f RMSE' % np.sqrt(testScore))


# Compare visually the outputs
print('---\nQuality evaluation:')
prediction = dnn.predict(
    testing_data,
    seq_len=training_settings["seq_length"]).flatten()


plt.parrallel_plot([np.concatenate([t.flatten() for t in testing_data.output]), prediction],
                   ["Ground truth", "Conv"],
                   "Neural network predictions vs ground truth")

print('--Done')
