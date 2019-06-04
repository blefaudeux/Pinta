#!/usr/bin/env python3

import numpy as np
from data_processing import plot as plt
from data_processing.load import load_folder, package_data
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

dnn = Conv(logdir='logs/' + settings.get_name(),
           input_size=INPUT_SIZE,
           hidden_size=training_settings["hidden_size"],
           filename='trained/' + settings.get_name() + '.pt')

# Load pre-computed normalization values
dnn.updateNormalization(training_settings)

if not dnn.valid():
    dnn.fit(training_data,
            testing_data,
            settings=training_settings,
            epoch=EPOCH,
            batch_size=BATCH_SIZE,
            self_normalize=False)
    dnn.save('trained/' + settings.get_name() + '.pt')


testScore = dnn.evaluate(
    testing_data,
    training_settings)

print('Final test Score: %.2f RMSE' % np.sqrt(testScore))


# Compare visually the outputs
print('---\nQuality evaluation:')
prediction = dnn.predict(
    testing_data,
    seq_len=training_settings["seq_length"]).flatten()

# Split the output sequence to re-align,
# ! need to take sequence length into account, offset
reference = []
splits = []
i = 0

for dataset in testing_data.output:
    reference.append(dataset[training_settings["seq_length"]-1:])
    i += reference[-1].shape[0]
    splits.append(i)

prediction = np.split(prediction, splits)

plt.parallel_plot(reference + prediction,
                  ["Ground truth" for _ in range(len(testing_data.output))] +
                  ["Conv" for _ in range(len(prediction))],
                  "Network predictions vs ground truth")

print('--Done')
