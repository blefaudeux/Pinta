#!/usr/bin/env python3

from data_processing.plot import multi_plot, speed_plot
from data_processing.load import load

# Load the full json and do some analysis
df = load('data/03_09_2016.json', clean_data=True, whiten_data=True)[0]

multi_plot(df, ['rudder_angle', 'wind_speed', 'wind_angle', 'boat_speed'],
           "test multi plot", "multi_plot", True)

# Test an effect of the rudder angle on speed
# TODO: Create a plot wrapper for this, correlation style
# df.plot(x=df['boat_speed'][:-1]-df['boat_speed'][1:],
#         y=df['rudder_angle'].iloc[:-1], style='*')

speed_plot(df, decimation=2, filename="speed_polar")
