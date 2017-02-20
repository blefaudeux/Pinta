#!/usr/local/bin/python3
import tensorflow as tf   # Bugfix in between Keras and TensorFlow
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model
tf.python.control_flow_ops = tf


class NN():

    def __init__(self):
        self.model = None
        self.valid = False

    def load(self, filename):
        try:
            self.model = load_model(filename)
            print("---\nNetwork {} loaded".format(filename))
            print(self.model.summary())
            return True

        except (ValueError, OSError, IOError) as e:
            print("Could not find or load existing NN")
            return False

    def save(self, name):
        self.model.save(name)

    def fit(self, train_in, train_out, iter=50, batch_size=1, verbose=2):
        self.model.fit(train_in, train_out, nb_epoch=iter,
                       batch_size=batch_size, verbose=verbose)
        self.valid = True


class SimpleNN(NN):

    def __init__(self, filename=None):
        # Load from trained NN if required
        if filename is not None:
            # TODO: Catch load failure ?
            self.valid = self.load(filename)

        else:
            # Else define the model
            hidden_neurons = 32
            self.model = Sequential()
            self.model.add(Dense(hidden_neurons, input_dim=3))
            self.model.add(Dense(hidden_neurons, activation='relu'))
            self.model.add(Dense(hidden_neurons, activation='relu'))
            self.model.add(Dense(1, activation='linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')
            print("Simple NN defined as follows\n" + str(self.model.summary()))


class MemoryNN(NN):

    def __init__(self, filename=None):
        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)
        else:
            hidden_neurons = 64
            self.model = Sequential()
            self.model.add(
                LSTM(input_dim=3, output_dim=hidden_neurons,
                     return_sequences=False))

            self.model.add(Dense(hidden_neurons, activation='relu'))
            self.model.add(Dense(hidden_neurons, activation='relu'))
            self.model.add(Dense(hidden_neurons, activation='relu'))
            self.model.add(
                Dense(1, input_dim=hidden_neurons, activation='linear'))

            self.model.compile(loss="mean_squared_error", optimizer="rmsprop")
