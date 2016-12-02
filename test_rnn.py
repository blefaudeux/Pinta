import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from data_processing.nmea2pandas import load_json
import data_processing.plot as plt

# Load the dataset
df = load_json('data/3_09_2016.json')
df['rudder_angle'] -= df['rudder_angle'].mean()
df = df.iloc[-6000:-2000].dropna()  # Last part, valid data

# Split in between training and test
training_ratio = 0.67   # Train on 2/3rds, test on the rest
train_size = int(len(df) * training_ratio)
test_size = len(df) - train_size
train, test = df.iloc[:train_size], df.iloc[train_size:len(df), :]

# Create lookback data window
# TODO: Ben. Allows for a time dependence of the predictions

# Create and fit Multilayer Perceptron model
train_inputs = np.array([train['wind_speed'].values, train['wind_angle'].values]).transpose()
train_output = train['boat_speed'].values
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_inputs, train_output, nb_epoch=100, batch_size=2, verbose=2)

# Estimate model performance
trainScore = model.evaluate(train_inputs, train_output, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, np.sqrt(trainScore)))

test_inputs = np.array([test['wind_speed'].values, test['wind_angle'].values]).transpose()
test_output = test['boat_speed'].values
testScore = model.evaluate(test_inputs, test_output, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, np.sqrt(testScore)))

# Compare visually the outputs :
pred = model.predict(test_inputs)
plt.parrallel_plot(test_output, pred, "Ref against predictions")
