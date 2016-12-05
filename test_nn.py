import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from data_processing.nmea2pandas import load_json
import data_processing.plot as plt

import tensorflow as tf   # Bugfix in between Keras and TensorFlow
tf.python.control_flow_ops = tf


# Load the dataset
df = load_json('data/3_09_2016.json')
df['rudder_angle'] -= df['rudder_angle'].mean()
df = df.iloc[-7000:-1000].dropna()  # Last part, valid data

plt.parrallel_plot([df['wind_speed'], df['boat_speed']],
                   ["Wind speed", "Boat speed"],
                   "Dataset plot")  # Split in between training and test

training_ratio = 0.67   # Train on 2/3rds, test on the rest
train_size = int(len(df) * training_ratio)
print("Training set is {} samples long".format(train_size))
test_size = len(df) - train_size
train, test = df.iloc[:train_size], df.iloc[train_size:len(df), :]

# Create lookback data window
# TODO: Ben. Allows for a time dependence of the predictions

# Create and fit Multilayer Perceptron model
train_inputs = np.array([train['wind_speed'].values, train['wind_angle'].values,
                         train['rudder_angle'].values]).transpose()

train_output = np.array(train['boat_speed'].values)

test_inputs = np.array([test['wind_speed'].values, test['wind_angle'].values,
                        test['rudder_angle'].values]).transpose()

test_output = np.array(test['boat_speed'].values)

#########################################################
# Super basic NN
name_simple = "simple_nn.hf5"

try:
    model_simple = load_model(name_simple)
    print("Network {} loaded".format(name_simple))
except (ValueError, OSError) as e:
    print("Could not find existing network, computing it on the fly\nThis may take time..")
    print('\n******\nTrain Simple NN...')

    model_simple = Sequential()
    model_simple.add(Dense(32, input_dim=3, activation='relu'))
    # model_simple.add(Dense(16))
    model_simple.add(Dense(1))

    model_simple.compile(loss='mean_squared_error', optimizer='adam')
    model_simple.fit(train_inputs, train_output, nb_epoch=100, batch_size=2, verbose=2)
    model_simple.save(name_simple)

# Estimate model performance
trainScore = model_simple.evaluate(train_inputs, train_output, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

testScore = model_simple.evaluate(test_inputs, test_output, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

#########################################################
# Inject LTSM to the mix:
name_lstm = "lstm_nn.hf5"

try:
    model_ltsm = load_model(name_lstm)
    print("Network {} loaded".format(name_lstm))
except (ValueError, OSError) as e:
    print("Could not find existing network, computing it on the fly\nThis may take time..")
    print('\n******\nTrain LSTM network...')

    model_ltsm = Sequential()
    model_ltsm.add(LSTM(output_dim=32, input_dim=3))
    # model_ltsm.add(LSTM(16))
    model_ltsm.add(Dense(1))

    model_ltsm.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Reshape inputs, timesteps must be in the training data
    train_inputs = np.reshape(train_inputs, (train_inputs.shape[0], 1, train_inputs.shape[1]))

    model_ltsm.fit(train_inputs, train_output, batch_size=1, nb_epoch=50, verbose=2)

    # Estimate model performance
    # trainScore = model_ltsm.evaluate(train_inputs, train_output, verbose=0)
    # print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

    # testScore = model_ltsm.evaluate(test_inputs, test_output, verbose=0)
    # print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

    model_ltsm.save(name_lstm)


# Compare visually the outputs :
print('Quality evaluation')
pred_simple = model_simple.predict(test_inputs).flatten()
test_inputs = np.reshape(test_inputs, (test_inputs.shape[0], 1, test_inputs.shape[1]))
pred_ltsm = model_ltsm.predict(test_inputs).flatten()

plt.parrallel_plot([test_output, pred_ltsm, pred_simple],
                   ["Ground truth", "LTSM", "Simple NN"],
                   "Testing neural network predictions against ground truth")

print('Done')
