#!/usr/local/bin/python3
import tensorflow as tf   # Bugfix in between Keras and TensorFlow
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential, load_model
from data_processing.nmea2pandas import load_json
from data_processing.split import split
import numpy as np
import gym
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.optimizers import Adam
tf.python.control_flow_ops = tf

# --------------------------------------------------
# Load the data on which to do the machine learning
df = load_json('data/31_08_2016.json', skip_zeros=True)
df['rudder_angle'] -= df['rudder_angle'].mean()
df = df.iloc[6000:-2000]

# Split in between training and test
training_ratio = 0.67
train_in, train_out, test_in, test_out = split(
    df,
    ['wind_speed', 'wind_angle', 'rudder_angle'],
    ['boat_speed'],
    training_ratio)

# --------------------------------------------------
# Load the NN simulating the boat
boat_nn = "simple_nn.hf5"

try:
    model_simple = load_model(boat_nn)
    print("Network {} loaded".format(boat_nn))
    print(model_simple.summary())

except (ValueError, OSError) as e:
    print("Could not find existing network, cannot continue")
    exit(0)

# --------------------------------------------------
# Learn how to steer..
# - keras-rl needs the model to inherit from gym
ENV_NAME = 'vpp-nn'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# - declare the keras-rl NN. Reuse a code sample for a start
nb_actions = 1
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Try to learn something..
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)

