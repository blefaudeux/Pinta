import numpy as np
from settings import Dataframe


def split(raw_data, settings):
    cat_in = settings["inputs"]
    cat_out = settings["outputs"]
    ratio = settings["training_ratio"]

    train_size = int(len(raw_data) * ratio)
    print("Training set is {} samples long".format(train_size))

    train, test = raw_data.iloc[:train_size], \
        raw_data.iloc[train_size:len(raw_data)]

    train_inputs = np.array([train[cat].values for cat in cat_in])
    test_inputs = np.array([test[cat].values for cat in cat_in])

    # Move samples to first dimension, makes more sense if output is 1d
    train_output = np.array([train[cat].values for cat in cat_out]).transpose()
    test_output = np.array([test[cat].values for cat in cat_out]).transpose()

    return Dataframe(train_inputs, train_output), Dataframe(test_inputs, test_output)
