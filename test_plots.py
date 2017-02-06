from data_processing.plot import multi_plot
from data_processing.load import load

# Load the full json and do some analysis
df = load('data/3_09_2016.json', clean_data=True)
multi_plot(df, ['rudder_angle', 'wind_speed', 'wind_angle'],
           "test multi plot", "multi_plot", True)

# Test an effect of the rudder angle on speed
# TODO: Create a plot wrapper for this, correlation style
# df.plot(x=df['boat_speed'][:-1]-df['boat_speed'][1:],
#         y=df['rudder_angle'].iloc[:-1], style='*')
