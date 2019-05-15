#!/usr/bin/env python3

import numpy as np
from data_processing import plot as plt
from data_processing.split import split
from data_processing.load import load
# from train.engine_rnn import ConvRNN
from train.engine_cnn import Conv
import settings


def angle_split(data):
    data['wind_angle_x'] = np.cos(np.radians(data['wind_angle']))
    data['wind_angle_y'] = np.sin(np.radians(data['wind_angle']))
    return data


training_settings = settings.get_defaults()

# Load the dataset + some data augmentation
datafile = 'data/03_09_2016.json'
raw_data, raw_data_aug = load(datafile, clean_data=True)

# Handle the angular coordinates discontinuity -> split x/y components
raw_data, raw_data_aug = angle_split(raw_data), angle_split(raw_data_aug)

# Small debug plot, have a look at the data
data_plot = training_settings["inputs"] + training_settings["outputs"]
plt.parrallel_plot([raw_data[i] for i in data_plot], data_plot, "Dataset plot")

# Split in between training and test
train_in, train_out, test_in, test_out = split(raw_data, training_settings)
train_in_r, train_out_r, test_in_r, test_out_r = split(
    raw_data_aug, training_settings)

train_in += train_in_r
train_out += train_out_r
test_in += test_in_r
test_out += test_out_r

# ConvRNN
BATCH_SIZE = training_settings["batch_size"]
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]),
              training_settings["seq_length"]]

print(f"Training on {len(train_in[0])} samples. Batch is {BATCH_SIZE}")
dnn = Conv(logdir='logs/conv',
           input_size=INPUT_SIZE,
           hidden_size=training_settings["hidden_size"],
           filename=training_settings["network_filename"])

if not dnn.valid():
    dnn.fit([train_in, train_out],
            [test_in, test_out],
            epoch=EPOCH,
            batch_size=BATCH_SIZE,
            seq_len=training_settings["seq_length"])
    dnn.save(training_settings["network_filename"])


testScore = dnn.evaluate(
    [test_in, test_out], seq_len=training_settings["seq_length"])
print('Final test Score: %.2f RMSE' % np.sqrt(testScore))


# Compare visually the outputs
print('---\nQuality evaluation:')
pred_simple = dnn.predict(
    [test_in, test_out],
    seq_len=training_settings["seq_length"]).flatten()

plt.parrallel_plot([test_out.flatten(), pred_simple],
                   ["Ground truth", "Conv"],
                   "Neural network predictions vs ground truth")

print('--Done')
