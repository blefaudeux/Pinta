#!python3
import numpy as np
from data_processing.split import split
from data_processing.load import load
from train.behaviour_gym import BoatNN
from rl.memory import SequentialMemory
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy
from rl.optimizers import Adam
import tensorflow as tf
tf.python.control_flow_ops = tf

# WIP - not working yet, that's expected

# Load the dataset
datafile = 'data/31_08_2016.json'
df = load(datafile, clean_data=True)

# Split in between training and test
inputs = ['wind_speed', 'wind_angle', 'rudder_angle']
training_ratio = 0.67
train_in, train_out, test_in, test_out = split(df, inputs,
                                               ['boat_speed'],
                                               training_ratio)

# --------------------------------------------------
# Learn how to steer..
# - load the boat environment
env = BoatNN()
env.configure('trained/simple_nn.hf5')

# - declare the keras-rl NN. Reuse a code sample for a start
nb_actions = 1
pilotNN = Sequential()
pilotNN.add(Flatten(input_shape=(1,) + env.observation_space.shape))
pilotNN.add(Dense(16))
pilotNN.add(Activation('relu'))
pilotNN.add(Dense(16))
pilotNN.add(Activation('relu'))
pilotNN.add(Dense(16))
pilotNN.add(Activation('relu'))
pilotNN.add(Dense(nb_actions))
pilotNN.add(Activation('linear'))
print("Declared Pilot NN:\n{}".format(pilotNN.summary()))

# - compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()

dqn = DDPGAgent(env=pilotNN, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                target_pilotNN_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Try to learn something..
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)
