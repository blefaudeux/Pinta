from data_processing.nmea2pandas import parse_nmea, save_json, load_json
from data_processing.plot import speed_plot, rudder_plot

# Parse the raw file and create a more compact and faster to read json
# df = parse_nmea("data/03_09_2016.nmea", 5.) # Added +5 to the wind angle
# save_json(df, 'data/3_09_2016.json')

# Load the json and do some analysis
df = load_json('data/3_09_2016.json')

# Rudder angle offset :
print("Average raw rudder angle: {:.2f}. Offsetting ".format(df['rudder_angle'].mean()))
df['rudder_angle'] -= df['rudder_angle'].mean()

# Value correlations
df.corr()

# Plotting the results
# speed_plot(df, 1)
# rudder_plot(df)

# Subselect the first part of the data
df = df.iloc[:5000]

print("Done")
