#!/usr/bin/env python3
import numpy as np
from data_processing import plot as plt
from data_processing.split import split
from data_processing.load import load
from train.behaviour import ConvRNN

# Load the dataset
datafile = 'data/31_08_2016.json'
raw_data = load(datafile, clean_data=True)

# Small debug plot, have a look at the data
data_plot = ['wind_angle', 'rudder_angle', 'boat_speed']
plt.parrallel_plot([raw_data[i] for i in data_plot], data_plot, "Dataset plot")

# Split in between training and test
inputs = ['wind_speed', 'wind_angle', 'rudder_angle']
outputs = ['boat_speed']
training_ratio = 0.67
train_in, train_out, test_in, test_out = split(raw_data, inputs,
                                               outputs, training_ratio)

# ConvRNN
CONV_SAVED = "trained/conv_rnn.hf5"
INPUT_SIZE = 3
LAYERS = 5
EPOCH = 20
BATCH_SIZE = 100
HIDDEN_SIZE = 90
crnn = ConvRNN(input_size=INPUT_SIZE,
               hidden_size=HIDDEN_SIZE,
               filename=CONV_SAVED,
               n_layers=LAYERS)

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
pred_simple = crnn.predict(test_in, batch_size=BATCH_SIZE).flatten()

plt.parrallel_plot([test_out.flatten(), pred_simple],
                   ["Ground truth", "Conv+RNN"],
                   "Neural network predictions vs ground truth")

print('--Done')
