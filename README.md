# Pinta Pilot
Pilot, machine learning, fitering and physics experiment

## Requirements
Everything is in python3. You can use *pip* to install dependencies, or your favorite package manager (brew, apt-get, whatever crap there is on windows)

Required libs:
* Theano or TensorFlow
* Keras
* h5py
* plotly

## Data
### Format
NMEA logs get converted in jsons, easier to parse afterwards to feed the learning. You can dump the raw logs in the /data folder, and run the convert_data.py script to update the json pool.

### Auto-cleaning tools
TODO: Bias, noise, coherency could be checked automatically. Some basic operations are already done through the clean_data flag, probably not enough


## Plots
There are a few wrappers to try to simplify typical plot calls (compare several traces for instance), nothing fancy, just to try to keep the code clean and concise. Generated plots are saved in the /plots subfolder. 
Visualisations for the NN could be added, probably a good idea (can also be added as renders when we optimize the networks)


## Neural nets - behaviour simulation
All the behaviour simulators inherit from the same NN class (defined in /train/behaviour), they can all basically be trained and predict output from inputs. All the trained networks can be saved under .hf5 files, in the /trained folder. This should simplify comprehensive testing of a lot of different NN architectures (with or without states, with different layer configurations, etc..).
An example of training is done through the test_vpp.py script, feel free to implement more networks as you see fit.


## Neural nets - reinforcement learning
WIP, should reuse [keras-rl](https://github.com/matthiasplappert/keras-rl) infrastructure, we need to wrap a behavioural network and 'maskerade' it as an [OpenAI-gym](https://gym.openai.com/) environment. This implies finding a proper cost function over time, not entirely trivial.


## Fusion
Another WIP, the idea here would be to properly filter some inputs (attitude, speed, position) to simplify the work for the subsequent NN


## GPU-based machine learning (nvidia graphics card only, theano backend)
You'll need to install Cuda and cuDNN. After this use 
> THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python3 myprogram.py
calls. If TensorFlow is used as a backend, and not Theano, edit your .keras/keras.json and change the backend accordingly.



**In any case, feel free to contribute, test, break things !**