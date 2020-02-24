#!/usr/bin/env python3

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from data_processing import plot as plt
from data_processing.load import get_train_test_list, load_folder, load_sets
from train.engine_cnn import Conv

training_settings = settings.get_defaults()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(training_settings["log"])

# Load the dataset + some data augmentation
data_list = load_sets(load_folder(Path('data')), training_settings)
training_data, testing_data = get_train_test_list(
    data_list, training_settings["training_ratio"], randomize=False)

# Get the total number of samples
n_training_samples = 0
[n_training_samples+s.inputs.shape[0] for s in training_data]

# ConvRNN
BATCH_SIZE = training_settings["batch_size"]
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]),
              training_settings["seq_length"]]

log.info(f"Training on {n_training_samples} samples. Batch is {BATCH_SIZE}")

dnn = Conv(logdir='logs/' + settings.get_name() + str(datetime.now()),
           input_size=INPUT_SIZE,
           hidden_size=training_settings["hidden_size"],
           filename='trained/' + settings.get_name() + '.pt',
           log_channel="DNN  ")

# Load pre-computed normalization values
dnn.update_normalization(training_settings)

if not dnn.valid:
    dnn.fit(training_data,
            testing_data,
            settings=training_settings,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            self_normalize=False)
    dnn.save('trained/' + settings.get_name() + '.pt')

log.info('Final test Score: %.2f RMSE' % np.sqrt(dnn.evaluate(
    testing_data,
    training_settings)))


# Compare visually the outputs
log.info('---\nQuality evaluation:')
prediction = dnn.predict(
    testing_data,
    seq_len=training_settings["seq_length"])

prediction = prediction.detach().cpu().numpy()

# Split the output sequence to re-align,
# ! need to take sequence length into account, offset
reference = []
splits = []
i = 0

for dataset in testing_data:
    reference.append(
        dataset.outputs[:-training_settings["seq_length"]+1].detach().cpu().numpy())
    i += reference[-1].shape[0]
    splits.append(i)

prediction = np.split(prediction, splits)

plt.parallel_plot(reference + prediction,
                  ["Ground truth" for _ in range(len(testing_data))] +
                  ["Conv" for _ in range(len(prediction))],
                  "Network predictions vs ground truth")

log.info('--Done')
