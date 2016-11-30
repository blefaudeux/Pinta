from data_processing.nmea2pandas import parse_nmea, save_json, load_json
from data_processing.plot import speed_plot, rudder_plot

# Parse the raw file and create a more compact and faster to read json
df = parse_nmea("data/03_09_2016.nmea")
save_json(df, 'data/3_09_2016.json')

# Load the json and do some analysis
# df = load_json('data/3_09_2016.json')

# Rudder angle offset :
print("Average raw rudder angle: {}".format(df['rudder_angle'].mean()))
df['rudder_angle'] -= df['rudder_angle'].mean()

# Plotting the results
speed_plot(df, 1)
rudder_plot(df)

print("Done")
