#!/usr/bin/python3
from data_processing.nmea2pandas import load_json
from data_processing.plot import scatter_plot
import numpy as np

if __name__ == '__main__':
    # Test the above
    df = load_json('../data/31_08_2016.json', skip_zeros=True)

    # Take the wind projection into account:
    projected_wind = df.wind_speed * np.sin(np.deg2rad(df.wind_angle))

    scatter_plot([df['boat_speed'].values, projected_wind.values],
                 ['Boat speed', 'Projected wind speed'],
                 'Boat speed vs Proj wind speed')