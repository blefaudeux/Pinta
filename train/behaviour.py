#!/usr/local/bin/python3
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv1D, Conv3D


class NN():
    def __init__(self):
        self._model = None  # We keep the actual NN private, safer
        self.valid = False

    def load(self, filename):
        try:
            self._model = load_model(filename)
            print("---\nNetwork {} loaded".format(filename))
            print(self._model.summary())
            return True

        except (ValueError, OSError, IOError) as e:
            print("Could not find or load existing NN")
            return False

    def save(self, name):
        self._model.save(name)

    def fit(self, train_in, train_out, epoch=50, batch_size=1, verbose=2):

        print("Training the network...")
        self._model.fit(
            train_in,
            train_out,
            epoch=epoch,
            batch_size=batch_size,
            verbose=verbose
        )
        self.valid = True
        print("... Done")

    def predict(self, inputs):
        return self._model.predict(inputs)

    def evaluate(self, inputs, outputs, verbose):
        # More settings could be exposed here. Needed ?
        return self._model.evaluate(inputs, outputs, verbose=verbose)


class ConvNN(NN):
    def __init__(self, filename=None):
        super(ConvNN, self).__init__()

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)
            if self.valid:
                return
            else:
                print(
                    "Could not load the specified net, computing it from scratch"
                )

        # Else define the _model
        hidden_neurons = 16
        nb_filters = 16
        filter_length = 10
        self._model = Sequential()
        self._model.add(
            Conv1D(
                filters=nb_filters,
                kernel_size=filter_length,
                activation='relu',
                input_shape=(None, 3)
            )
        )
        self._model.add(Activation('sigmoid'), input_dim=nb_filters)
        self._model.add(MaxPooling1D(pool_length=4))
        self._model.add(
            Convolution1D(nb_filter=32, filter_length=8, border_mode='valid')
        )
        self._model.add(Activation('sigmoid'))
        self._model.add(MaxPooling1D(pool_length=4))
        self._model.add(Flatten())
        self._model.add(Dense(hidden_neurons))
        self._model.add(Dense(1, input_dim=hidden_neurons, activation='linear'))
        self._model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            sample_weight_mode='temporal'
        )
        print(
            "Conv NN defined as follows\n \
          {}".format(str(self._model.summary()))
        )


class MemoryNN(NN):
    def __init__(self, filename=None):
        super(MemoryNN, self).__init__()

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)

        if self.valid:
            return

        # Else define the _model
        hidden_neurons = 32
        self._model = Sequential()
        self._model.add(Dense(hidden_neurons, input_dim=3, activation='relu'))
        self._model.add(LSTM(hidden_neurons, return_sequences=False))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(1, input_dim=hidden_neurons, activation='linear'))

        self._model.compile(loss="mean_squared_error", optimizer="rmsprop")
