from data_processing.nmea2pandas import load_json
from data_processing.whitening import whiten_angle


def load(filename, clean_data=True, whiten_data=True):
    df = load_json(filename, skip_zeros=True)

    # Fix a possible offset in the rudder angle sensor
    if clean_data:
        df['rudder_angle'] -= df['rudder_angle'].mean()

    # Whiten the data, in that the boat supposedely goes at the same speed port and starboard
    if whiten_data:
        df_white_angle = whiten_angle(df)
    else:
        df_white_angle = None

    return [df, df_white_angle]
