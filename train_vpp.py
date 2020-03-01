#!/usr/bin/env python3

import logging
from datetime import datetime
from pathlib import Path

import numpy as np

import settings
from data_processing import plot as plt
from data_processing.load import load_folder, load_sets
from data_processing.training_set import TrainingSetBundle
from train.engine_cnn import Conv

training_settings = settings.get_defaults()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(training_settings["log"])

# Load the dataset + some data augmentation
data_list = load_sets(load_folder(Path('data')), training_settings)
training_bundle = TrainingSetBundle(data_list)

# TODO: Ben - pass the transforms here

#  ConvRNN
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]),
              training_settings["seq_length"]]

log.info("Training on {} samples. Batch is {}".format(
    len(training_bundle), training_settings["batch_size"]))

dnn = Conv(logdir='logs/' + settings.get_name() + str(datetime.now()),
           input_size=INPUT_SIZE,
           hidden_size=training_settings["hidden_size"],
           filename='trained/' + settings.get_name() + '.pt',
           log_channel="DNN  ")

# Load pre-computed normalization values
trainer, tester = training_bundle.get_dataloaders(
    training_settings["training_ratio"],
    training_settings["seq_length"],
    training_settings["batch_size"],
    shuffle=True)

if not dnn.valid:
    dnn.fit(trainer,
            tester,
            settings=training_settings,
            epochs=EPOCH)
    dnn.save('trained/' + settings.get_name() + '.pt')

log.info('Final test Score: %.2f RMSE' % np.sqrt(dnn.evaluate(
    tester,
    training_settings)))


# Compare visually the outputs
log.info('---\nQuality evaluation:')
prediction = dnn.predict(
    tester,
    seq_len=training_settings["seq_length"])

prediction = prediction.detach().cpu().numpy()

# Split the output sequence to re-align,
# ! need to take sequence length into account, offset
reference = []
splits = []
i = 0

for dataset in tester:
    reference.append(
        dataset.outputs[:-training_settings["seq_length"]+1]
        .detach()
        .cpu()
        .numpy())
    i += reference[-1].shape[0]
    splits.append(i)

prediction = np.split(prediction, splits)

plt.parallel_plot(reference + prediction,
                  ["Ground truth" for _ in range(len(tester))] +
                  ["Conv" for _ in range(len(prediction))],
                  "Network predictions vs ground truth")

log.info('--Done')
