#!/usr/bin/env python3

import numpy as np
from data_processing import plot as plt
from data_processing.split import split
from data_processing.load import load
from train.behaviour import ConvRNN


def angle_split(data):
    data['wind_angle_x'] = np.cos(np.radians(data['wind_angle']))
    data['wind_angle_y'] = np.sin(np.radians(data['wind_angle']))
    return data


# Load the dataset
datafile = 'data/03_09_2016.json'
raw_data, raw_data_reversed = load(datafile, clean_data=True)
INPUTS = ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle']
OUTPUTS = ['boat_speed']

# Handle the angular coordinates discontinuity -> split x/y components
raw_data = angle_split(raw_data)
raw_data_reversed = angle_split(raw_data_reversed)

# Small debug plot, have a look at the data
data_plot = INPUTS + OUTPUTS
plt.parrallel_plot([raw_data[i] for i in data_plot], data_plot, "Dataset plot")

# Split in between training and test
training_ratio = 0.8
train_in, train_out, test_in, test_out = split(raw_data, INPUTS,
                                               OUTPUTS, training_ratio)

train_in_r, train_out_r, test_in_r, test_out_r = split(raw_data_reversed, INPUTS,
                                                       OUTPUTS, training_ratio)


train_in += train_in_r
train_out += train_out_r
test_in += test_in_r
test_out += test_out_r

# ConvRNN
CONV_SAVED = "trained/conv_rnn.torch"
INPUT_SIZE = len(INPUTS)
GRU_LAYERS = 6
EPOCH = 100
BATCH_SIZE = 1000
HIDDEN_SIZE = 60
crnn = ConvRNN(logdir='logs/gru6conv60',
               input_size=INPUT_SIZE,
               hidden_size=HIDDEN_SIZE,
               filename=CONV_SAVED,
               n_gru_layers=GRU_LAYERS)

if not crnn.valid:
    crnn.fit([train_in, train_out],
             [test_in, test_out],
             epoch=EPOCH,
             batch_size=BATCH_SIZE)
    crnn.save(CONV_SAVED)

trainScore = crnn.evaluate([train_in, train_out])
print('Train Score: %.2f RMSE' % np.sqrt(trainScore))

testScore = crnn.evaluate([test_in, test_out])
print('Test Score: %.2f RMSE' % np.sqrt(testScore))


# Compare visually the outputs
print('---\nQuality evaluation:')
pred_simple = crnn.predict(
    [test_in, test_out], batch_size=BATCH_SIZE).flatten()

plt.parrallel_plot([test_out.flatten(), pred_simple],
                   ["Ground truth", "Conv+RNN"],
                   "Neural network predictions vs ground truth")

print('--Done')
