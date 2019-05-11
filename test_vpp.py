#!/usr/bin/env python3

import numpy as np
from data_processing import plot as plt
from data_processing.split import split
from data_processing.load import load
# from train.engine_rnn import ConvRNN
from train.engine_cnn import Conv
from constants import INPUTS, OUTPUTS, NN_FILENAME, HIDDEN_SIZE, SEQ_LEN, INPUT_SIZE


def angle_split(data):
    data['wind_angle_x'] = np.cos(np.radians(data['wind_angle']))
    data['wind_angle_y'] = np.sin(np.radians(data['wind_angle']))
    return data


# Load the dataset + some data augmentation
datafile = 'data/03_09_2016.json'
raw_data, raw_data_aug = load(datafile, clean_data=True)

# Handle the angular coordinates discontinuity -> split x/y components
raw_data, raw_data_aug = angle_split(
    raw_data), angle_split(raw_data_aug)

# Small debug plot, have a look at the data
data_plot = INPUTS + OUTPUTS
plt.parrallel_plot([raw_data[i] for i in data_plot], data_plot, "Dataset plot")

# Split in between training and test
TRAINING_RATIO = 0.9
train_in, train_out, test_in, test_out = split(raw_data, INPUTS,
                                               OUTPUTS, TRAINING_RATIO)

train_in_r, train_out_r, test_in_r, test_out_r = split(raw_data_aug, INPUTS,
                                                       OUTPUTS, TRAINING_RATIO)

train_in += train_in_r
train_out += train_out_r
test_in += test_in_r
test_out += test_out_r

# ConvRNN
GRU_LAYERS = 2
EPOCH = 1
BATCH_SIZE = 5000

print(f"Training on {len(train_in[0])} samples. Batch is {BATCH_SIZE}")

# dnn = ConvRNN(logdir='logs/crnn',
#               input_size=INPUT_SIZE,
#               hidden_size=HIDDEN_SIZE,
#               filename=CONV_SAVED,
#               n_gru_layers=GRU_LAYERS)

# if not dnn.valid:
#     dnn.fit([train_in, train_out],
#             [test_in, test_out],
#             epoch=EPOCH,
#             batch_size=BATCH_SIZE)
#     dnn.save(CONV_SAVED)

dnn = Conv(logdir='logs/conv',
           input_size=INPUT_SIZE,
           hidden_size=HIDDEN_SIZE,
           filename=NN_FILENAME)

if not dnn.valid():
    dnn.fit([train_in, train_out],
            [test_in, test_out],
            epoch=EPOCH,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN)
    dnn.save(NN_FILENAME)


testScore = dnn.evaluate([test_in, test_out], seq_len=SEQ_LEN)
print('Final test Score: %.2f RMSE' % np.sqrt(testScore))


# Compare visually the outputs
print('---\nQuality evaluation:')
pred_simple = dnn.predict(
    [test_in, test_out], seq_len=SEQ_LEN).flatten()

plt.parrallel_plot([test_out.flatten(), pred_simple],
                   ["Ground truth", "Conv"],
                   "Neural network predictions vs ground truth")

print('--Done')
