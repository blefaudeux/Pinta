import plotly.graph_objs as go
import plotly.offline as py
import pynmea2 as nmea
import numpy as np
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
    'IIXTE': 'CrossTrackError',
    'IIRSA': 'RudderAngle'
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
ws = []

for ts in data['Speed'].keys():
    if ts in data['WindTrue'].keys():
        try:
            twa.append((float(data['Speed'][ts][0]) - float(data['WindTrue'][ts][0]) + 180) % 360. - 180.)
            boat_speed.append(float(data['Speed'][ts][4]))
            ws.append(float(data['WindTrue'][ts][4]))
        except ValueError:
            # Corrupted data, once again
            pass

# Display some data
print("\nPlotting data")
decimation = 10  # Stupid decimation to begin with

# - raw polar plot
# we need to use x,y plots, plotly polar plots are broken
twa_rad = np.radians(twa[::decimation])
labels = ['TWA: {}d \n Boat speed: {}kt'.format(w, b) for w, b in zip(twa[::decimation], boat_speed[::decimation])]

speed = go.Scattergl(
    x=boat_speed[::decimation] * np.sin(twa_rad),
    y=boat_speed[::decimation] * np.cos(twa_rad),
    mode='markers',
    name='Boat speed',
    marker=dict(
        size=8,
        line=dict(width=1),
        color=ws[::decimation],
        colorscale='Portland',
        showscale=True,
        colorbar=go.ColorBar(
            title="Wind speed"
        ),
        opacity=0.7
    ),
    text=labels
)

traces = [speed]

layout = go.Layout(
    title='Speed vs TWA',
    orientation=90,
    autosize=False,
    width=1000,
    height=1000,
    hovermode='closest',
    plot_bgcolor='rgb(223, 223, 223)'
)
fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='speed.html', auto_open=False)
