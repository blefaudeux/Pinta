import pandas as pd
import pynmea2 as nmea
import time
import json

"""
    Parse linear NMEA logs, return a Pandas dataframe.
    Extra functions to load/save json files to speed up subsequent uses.
"""


# The fields that we know of, and which can be parsed into the dataframe
nmea_fields = {
    'IIVHW': 'Speed',
    'IIVLW': 'Log',
    'IIDPT': 'DepthSetup',
    'IIDBT': 'Depth',
    'IIMTW': 'WaterTemp',
    'IIVWR': 'ApparentWind',
    'IIMWD': 'WindTrue',
    'IIVWT': 'WindTrueLR',
    'IIMTA': 'AirTemp',
    'IIHDG': 'HeadingMag',
    'IIHDM': 'HeadingMagPrecise',
    'IIHDT': 'HeadingTrue',
    'IIZDA': 'Time',
    'IIGLL': 'Pose',
    'IIVGT': 'BottomHeading',
    'IIXTE': 'CrossTrackError',
    'IIRSA': 'RudderAngle'
}


def parse_nmea(filepath, wind_bias=0):
    f = open(filepath)
    data = {}
    for key in nmea_fields:
        data[nmea_fields[key]] = {}

    # Parse everything & dispatch in the appropriate categories
    skipped_fields = {}
    timestamp = None
    print("\nParsing the NMEA file {}".format(filepath))
    print("Wind direction bias: {}".format(wind_bias))
    for line in f:
        try:
            sample = nmea.parse(line)
            key = sample.identifier()[:-1]

            if key == 'IIZDA':
                timestamp = time.mktime(sample.datetime.timetuple())

            try:
                if timestamp is not None:
                    # One measurement per timestamp, for now
                    data[nmea_fields[key]][timestamp] = sample.data

            except KeyError:
                # We discard this field for now
                if key not in skipped_fields.keys():
                    print("Unknown field: {}".format(sample.identifier()[:-1]))
                    skipped_fields[key] = 1

        except (nmea.ParseError, nmea.nmea.ChecksumError, TypeError) as e:
            # Corrupted data, skip
            pass

    # Reorganize for faster processing afterwards, save in a pandas dataframe
    print("\nReorganizing data per timestamp")
    wa = []
    wa_index = []
    for ts in data['Speed'].keys():
        if ts in data['WindTrue'].keys():
            try:
                wa.append((float(data['Speed'][ts][0]) - float(data['WindTrue'][ts][0]) + wind_bias + 180) % 360.
                          - 180.)
                wa_index.append(ts)

            except ValueError:
                # Corrupted data, skip
                pass

    dataframe = {
        'wind_angle': pd.Series(wa, index=wa_index),
        'boat_speed': pd.Series([float(data['Speed'][ts][4]) for ts in data['Speed'].keys()],
                                index=data['Speed'].keys()),
        'wind_speed': pd.Series([float(data['WindTrue'][ts][4]) for ts in data['WindTrue'].keys()],
                                index=data['WindTrue'].keys()),
        'rudder_angle': pd.Series([float(data['RudderAngle'][ts][0]) for ts in data['RudderAngle'].keys()],
                                  index=data['RudderAngle'].keys())
    }

    return pd.DataFrame(dataframe)


def save_json(df, filename):
    with open(filename, 'w') as outfile:
        json.dump(df.to_json(), outfile)


def load_json(filename, skip_zeros=True):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    df = pd.read_json(data)
    return df if not skip_zeros else df[df.boat_speed > 0].dropna()
