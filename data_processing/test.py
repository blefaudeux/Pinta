from nmea2pandas import parse_nmea, save_json, load_json
from plot import speed_plot, rudder_plot

# df = parse_file("../data/03_09_2016.nmea")
df = load_json('3_09_2016.json')
save_json(df, '3_09_2016.json')

speed_plot(df, 1)
rudder_plot(df)

print("Done")
