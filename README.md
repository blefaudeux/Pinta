# Pinta Pilot
Pilot, machine learning, fitering and physics experiment

## Requirements
Everything is in python3. You can use *pip* to install dependencies, or your favorite package manager. One option is simply (preferably from a virtual environment)

`pip install -r requirements.txt`

## Data
### Format
NMEA logs get converted in jsons, easier to parse afterwards to feed the learning. You can dump the raw logs in the /data folder, and run the convert_data.py script to update the json pool.

### Auto-cleaning tools
Bias, noise, coherency could be checked automatically. Some basic operations are already done through the clean_data flag, probably not enough


## Plots
There are a few wrappers to try to simplify typical plot calls (compare several traces for instance), nothing fancy, just to try to keep the code clean and concise. Generated plots are saved in the /plots subfolder.
Visualisations for the NN could be added, probably a good idea (can also be added as renders when we optimize the networks)


## Neural nets - behaviour simulation
All the behaviour simulators inherit from the same NN class (defined in /train/behaviour), they can all basically be trained and predict output from inputs. All the trained networks can be saved under .pt files, in the /trained folder. This should simplify comprehensive testing of a lot of different NN architectures (with or without states, with different layer configurations, etc..).
An example of training is done through the train_vpp.py script, feel free to implement more networks as you see fit.


## Neural nets - reinforcement learning
WIP, should reuse existing pytorch/RL infrastructure, we need to wrap a behavioural network and 'maskerade' it as an [OpenAI-gym](https://gym.openai.com/) environment. This implies finding a proper cost function over time, not entirely trivial.


## GPU-based machine learning (nvidia graphics card only for now)
You'll need to install Cuda and cuDNN. After this the ./train_vpp.py script should automatically pick up the GPU. If you're seeing a memory error, well, that probably means that you're low on memory on the GPU with these settings. A batch gradient accumulation could be done automatically in that case, not done right now



**In any case, feel free to contribute, test, break things !**
