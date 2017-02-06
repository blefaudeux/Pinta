import numpy as np


def split(raw_data, cat_in, cat_out, ratio):
    train_size = int(len(raw_data) * ratio)
    print("Training set is {} samples long".format(train_size))

    train, test = raw_data.iloc[:train_size], \
        raw_data.iloc[train_size:len(raw_data), :]

    # Create and fit Multilayer Perceptron model
    train_inputs = np.array([train[cat].values for cat in cat_in]).transpose()
    train_output = np.array([train[cat].values for cat in cat_out]).transpose()

    test_inputs = np.array([test[cat].values for cat in cat_in]).transpose()
    test_output = np.array([test[cat].values for cat in cat_out]).transpose()

    return train_inputs, train_output, test_inputs, test_output