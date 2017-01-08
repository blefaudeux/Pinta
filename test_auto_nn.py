import numpy as np
import tensorflow as tf   # Bugfix in between Keras and TensorFlow
from keras.models import load_model

from data_processing.nmea2pandas import load_json

tf.python.control_flow_ops = tf

# --------------------------------------------------
# Load the data on which to do the machine learning
df = load_json('data/31_08_2016.json', skip_zeros=True)
df['rudder_angle'] -= df['rudder_angle'].mean()
df = df.iloc[6000:-2000]

# Split in between training and test
training_ratio = 0.67
train_size = int(len(df) * training_ratio)
print("Training set is {} samples long".format(train_size))
test_size = len(df) - train_size
train, test = df.iloc[:train_size], df.iloc[train_size:len(df), :]

# Create and fit Multilayer Perceptron model
train_inputs = np.array([train['wind_speed'].values, train['wind_angle'].values,
                         train['rudder_angle'].values]).transpose()

train_output = np.array(train['boat_speed'].values)

test_inputs = np.array([test['wind_speed'].values, test['wind_angle'].values,
                        test['rudder_angle'].values]).transpose()

test_output = np.array(test['boat_speed'].values)

# --------------------------------------------------
# Load the NN simulating the boat
boat_nn = "simple_nn.hf5"

try:
    model_simple = load_model(boat_nn)
    print("Network {} loaded".format(boat_nn))

except (ValueError, OSError) as e:
    print("Could not find existing network, cannot continue")
    exit(0)

# --------------------------------------------------
# Learn how to steer..

#TODO: Ben
