from nmea2pandas import parse_file
from plot import speed_plot, rudder_plot

df = parse_file("../data/03_09_2016.nmea")
speed_plot(df)
rudder_plot(df)

print("Done")
