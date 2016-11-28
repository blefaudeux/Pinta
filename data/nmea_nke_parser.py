import plotly.graph_objs as go
import plotly.offline as py
import pynmea2 as nmea
import time

filepath = "03_09_2016.nmea"

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
    'IIXTE': 'CrossTrackError'
}

f = open(filepath)
data = {}
for key in nmea_fields:
    data[nmea_fields[key]] = {}

# Parse everything & dispatch in the appropriate categories
skipped_fields = {}
timestamp = None
print("\nParsing the NMEA file {}".format(filepath))
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

# Reorganize for faster processing afterwards :
print("\nReorganizing data per timestamp")
boat_speed = []
twa = []

for ts in data['Speed'].keys():
    if ts in data['WindTrue'].keys():
        twa.append(float(data['WindTrue'][ts][0]))
        boat_speed.append(data['Speed'][ts][4])

# Display some data
print("\nPlotting data")
decimation = 10  # Stupid decimation to begin with 

# - raw polar plot
speed = go.Scatter(
    r=boat_speed[:decimation:],
    t=twa[:decimation:],
    mode='markers',
    name='Boat speed',
    marker=dict(
        line=dict(
            width=1,
            color='#404040')
    )
)

traces = [speed]
layout = go.Layout(
    title='Speed vs TWA',
    orientation=-90,
    font=dict(
        size=15
    ),
    plot_bgcolor='rgb(223, 223, 223)',
    angularaxis=dict(
        tickcolor='rgb(253,253,253)',
        range=[0, 360]
    ),
    xaxis=dict(
        range=[0, 20]
    ),
    yaxis=dict(
        range=[0, 20]
    )
)
fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='speed.html', auto_open=False)
