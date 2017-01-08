from data_processing.nmea2pandas import load_json
from data_processing.plot import *

# Parse the raw file and create a more compact and faster to read json
# df = parse_nmea("data/03_09_2016.nmea", 5.) # Added +5 to the wind angle
# save_json(df, 'data/3_09_2016.json')

# Load the full json and do some analysis
df = load_json('data/3_09_2016.json')

# Rudder angle offset :
print("Average raw rudder angle: {:.2f}. Offsetting ".format(df['rudder_angle'].mean()))
df['rudder_angle'] -= df['rudder_angle'].mean()

# Subselect the last part of the data
df = df.iloc[-6000:-2000]

multi_plot(df, ['rudder_angle', 'wind_speed'],"test multi plot", "multi_plot", True)

# Test an effect of the rudder angle on speed
# df.plot(x=df['boat_speed'][:-1]-df['boat_speed'][1:], y=df['rudder_angle'].iloc[:-1], style='*')
#
# # Plotting the results
# speed_plot(df, 1)
# rudder_plot(df)


print("Done")
