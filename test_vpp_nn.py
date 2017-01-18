#!/usr/local/bin/python3
import numpy as np
import tensorflow as tf   # Bugfix in between Keras and TensorFlow
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential, load_model

import data_processing.plot as plt
from data_processing.split import split
from data_processing.load import load

tf.python.control_flow_ops = tf

# Load the dataset
datafile = 'data/31_08_2016.json'
df = load(datafile, clean_data=True)
df = df.iloc[6000:-4000]

# Small debug plot, have a look at the data
inputs = ['wind_speed', 'wind_angle', 'rudder_angle']
plt.parrallel_plot([df[i] for i in inputs], inputs, "Dataset plot")

# Split in between training and test
training_ratio = 0.67
train_in, train_out, test_in, test_out = split(df, inputs,
                                               ['boat_speed'], 
                                               training_ratio)

#########################################################
# Super basic NN
name_simple = "trained/simple_nn.hf5"

try:
    model_simple = load_model(name_simple)
    print("---\nNetwork {} loaded".format(name_simple))
    print(model_simple.summary())

except (ValueError, OSError, IOError) as e:
    print("Could not find existing network, computing it on the fly\nThis may take time..")
    print('\n******\nTrain Simple NN...')

    model_simple = Sequential()
    model_simple.add(Dense(8, input_dim=3))
    model_simple.add(Dense(8, activation='relu'))
    model_simple.add(Dense(8, activation='relu'))
    model_simple.add(Dense(8, activation='relu'))
    model_simple.add(Dense(1, activation='linear'))
    print("Simple NN model, dense\n" + str(model_simple.summary()))

    model_simple.compile(loss='mean_squared_error', optimizer='adam')
    model_simple.fit(train_in, train_out, nb_epoch=50, batch_size=1, verbose=2)
    model_simple.save(name_simple)

# Estimate model performance
trainScore = model_simple.evaluate(train_in, train_out, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

testScore = model_simple.evaluate(test_in, test_out, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

#########################################################
# Inject LTSM to the mix:
name_lstm = "trained/lstm_nn.hf5"
hidden_neurons = 64
train_inputs_ltsm = np.reshape(train_in, (train_in.shape[0], 1, train_in.shape[1]))
test_inputs_ltsm = np.reshape(test_in, (test_in.shape[0], 1, test_in.shape[1]))

try:
    model_ltsm = load_model(name_lstm)
    print("---\nNetwork {} loaded".format(name_lstm))
    print(model_ltsm.summary())

except (ValueError, OSError, IOError) as e:
    print("Could not find existing LSTM network, computing it on the fly\nThis may take time..")
    print('\n******\nTrain LSTM network...')

    model_ltsm = Sequential()
    model_ltsm.add(LSTM(input_dim=3, output_dim=hidden_neurons, return_sequences=False))
    model_ltsm.add(Dense(hidden_neurons, activation='relu'))
    model_ltsm.add(Dense(hidden_neurons, activation='relu'))
    model_ltsm.add(Dense(hidden_neurons, activation='relu'))
    model_ltsm.add(Dense(hidden_neurons, activation='relu'))
    model_ltsm.add(Dense(1, input_dim=hidden_neurons, activation='linear'))
    model_ltsm.compile(loss="mean_squared_error", optimizer="rmsprop")
    print("LSTM-based RNN\n" + str(model_ltsm.summary()))

    # Reshape inputs, timesteps must be in the training data
    model_ltsm.fit(train_inputs_ltsm, train_out, nb_epoch=20, verbose=2)
    model_ltsm.save(name_lstm)

# Estimate model performance - arbitrary metric (LS probably)
trainScore = model_ltsm.evaluate(train_inputs_ltsm, train_out, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

testScore = model_ltsm.evaluate(test_inputs_ltsm, test_out, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# Compare visually the outputs
print('---\nQuality evaluation:')
pred_simple = model_simple.predict(test_in).flatten()
pred_ltsm = model_ltsm.predict(test_inputs_ltsm).flatten()

plt.parrallel_plot([test_out, pred_ltsm, pred_simple],
                   ["Ground truth", "LTSM", "Simple NN"],
                   "Testing neural network predictions against ground truth")

print('--Done')
