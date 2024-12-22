# Pinta  ![Basic Checks](https://github.com/blefaudeux/Pinta/workflows/Python%20package/badge.svg)
Pilot, machine learning, fitering and physics experiment

## Requirements & install
Everything is in python3. You can use *pip* to install dependencies, or your favorite package manager. One option is simply (preferably from a virtual environment)

Install all dependencies:
`pip install -r requirements.txt`

Install this repo
`pip install -e .`

## Data
### Format
NMEA logs get converted in jsons, easier to parse afterwards to feed the learning. You can dump the raw logs in the /data folder, and run the convert_data.py script to update the json pool.

### Auto-cleaning tools
Bias, noise, coherency could be checked automatically. Some basic operations are already done through the clean_data flag, probably not enough

## Plots
There are a few wrappers to try to simplify typical plot calls (compare several traces for instance), nothing fancy, just to try to keep the code clean and concise. Generated plots are saved in the /plots subfolder.
Visualisations for the NN could be added, probably a good idea (can also be added as renders when we optimize the networks)
- Plot a given file:
`./plot_data_file.py {pathToDataFile.json} `

![Experimental polar plot](../master/ressources/polar_experimental.png?raw=true "Experimental polar plot")
![Experimental trace](../master/ressources/trace.png?raw=true "Experimental trace")

## Neural nets - behaviour simulation
All the models inherit from the same NN class (defined in /pinta/model/model_base), irrespective of their architecture they can all basically be trained and predict output from inputs. All the trained networks can be saved as torchscript (.pt) files, in the /trained folder. This should simplify comprehensive testing of a lot of different NN architectures (with or without states, with different layer configurations, etc..).

An example of training is done through the *train_vpp.py* script, feel free to implement more networks as you see fit.

- Train a new model, given the settings in *settings.py*:
`./train_vpp.py --settings_path settings/settings_mini_polar.json --data_path data/ --parallel --amp `

This saves the resulting model, and produces a predicted curve against the ground truth, similar to the attached example. Noteworthy extra arguments are `--parallel` to load data files on multiple cores (useful if dataset spread over multiple files), and `--amp` to use automatic mixed precision (way faster training on somewhat recent GPUs).

![Prediction vs. Ground truth example](../master/ressources/evaluation.jpg?raw=true "Prediction vs. Ground truth example")


- Generate a polar plot from a trained model:
`./polar_from_vpp.py --model_path={pathToTrainedModel.pt}`

This does not use the whole model capacity, but can be useful to check that the results are meaningful

![Simulated polar](../master/ressources/polar_eval.jpg?raw=true "Simulated polar")
