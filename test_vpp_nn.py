#!/usr/local/bin/python3
import numpy as np
from data_processing import plt, split, load
from train.behaviour import SimpleNN, MemoryNN

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

# Super basic NN, FW stateless NN
name_simple = "trained/simple_nn.hf5"
snn = SimpleNN(name_simple)
if not snn.valid:
    snn.fit(train_in, train_out, nb_epoch=50, batch_size=1, verbose=2)
    snn.save(name_simple)

trainScore = snn.model.evaluate(train_in, train_out, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

testScore = snn.model.evaluate(test_in, test_out, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# Test a more complex NN, LSTM
train_inputs_ltsm = np.reshape(
    train_in, (train_in.shape[0], 1, train_in.shape[1]))
test_inputs_ltsm = np.reshape(test_in, (test_in.shape[0], 1, test_in.shape[1]))

name_lstm = "trained/lstm_nn.hf5"
mnn = MemoryNN(name_lstm)
if not mnn.valid:
    mnn.fit(train_inputs_ltsm, train_out, nb_epoch=20, verbose=2)
    mnn.save(name_lstm)

trainScore = mnn.model.evaluate(train_inputs_ltsm, train_out, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

testScore = mnn.model.evaluate(test_inputs_ltsm, test_out, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# Compare visually the outputs
print('---\nQuality evaluation:')
pred_simple = snn.model.predict(test_in).flatten()
pred_ltsm = mnn.model.predict(test_inputs_ltsm).flatten()

plt.parrallel_plot([test_out.flatten(), pred_ltsm, pred_simple],
                   ["Ground truth", "LTSM", "Simple NN"],
                   "Neural network predictions vs ground truth")

print('--Done')
