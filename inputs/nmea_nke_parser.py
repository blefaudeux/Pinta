import pynmea2 as nmea
import numpy as np

filepath = "../data/03_09_2016.nmea"

nmea_fields = {
    'IIVHW': 'Speed',
    'IIVLW': 'Log',
    'IIDPT': 'DepthSetup',
    'IIDBT': 'Depth',
    'IIMTW': 'WaterTemp',
    'IIVWR': 'ApparentWind',
    'IIMWD': 'TrueWind',
    'IIVWT': 'TrueWindLR',
    'IIMTA': 'AirTemp',
    'IIHDG': 'HeadingMag',
    'IIHDM': 'HeadingMagPrecise',
    'IIHDT': 'HeadingTrue',
    'IIZDA': 'Time',
    'IIGLL': 'Pose',
    'IIVGT': 'BottomHeading',
    'IIXTE': 'CrossTrackError'
}

f = open(filepath)
data = {}
for key in nmea_fields:
    data[nmea_fields[key]] = []

# Parse everything & dispatch in the appropriate categories
skipped_fields = {}
for line in f:
    try:
        sample = nmea.parse(line)
        key = sample.identifier()[:-1]

        try:
            data[nmea_fields[key]].append(sample.data)

        except KeyError:
            # We discard this field for now
            if key not in skipped_fields.keys():
                print("Skipped field: {}".format(sample.identifier()[:-1]))
                skipped_fields[key] = 1
            pass

    except (nmea.ParseError, nmea.nmea.ChecksumError) as e:
        # Corrupted data, skip
        pass

# Display some data
#TODO: Ben



