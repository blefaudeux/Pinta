#!/usr/bin/env python3

from data_processing.plot import multi_plot, speed_plot
from data_processing.load import load
from pathlib import Path

# Load the full json and do some analysis
df = load(Path('data/03_09_2016.json'), clean_data=True)

# Purely sequential plot
multi_plot(df, ['rudder_angle', 'wind_speed', 'wind_angle', 'boat_speed'],
           "test multi plot", "multi_plot", True)

# Polar plot
speed_plot(df, decimation=2, filename="speed_polar")
