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

# Reorganize for faster processing afterwards, align with timestamp:
print("\nReorganizing data per timestamp")
bs = []
wa = []
ws = []
ra = []

for ts in data['Speed'].keys():
    if ts in data['WindTrue'].keys() and ts in data['Speed'].keys() and ts in data['RudderAngle'].keys():
        try:
            wa.append((float(data['Speed'][ts][0]) - float(data['WindTrue'][ts][0]) + 180) % 360. - 180.)
            bs.append(float(data['Speed'][ts][4]))
            ws.append(float(data['WindTrue'][ts][4]))
            ra.append(float(data['RudderAngle'][ts][0]))

        except ValueError:
            # Corrupted data or no matching data
            pass

# Display some data
print("\nPlotting data")
decimation = 2  # Stupid decimation to begin with

# - raw polar plot
# we need to use x,y plots, plotly polar plots are broken
twa_rad = np.radians(wa[::decimation])
labels = ['Wind: {}deg - {}kt \nBoat speed: {}kt * Rudder: {}deg'.format(wa, ws, b, r) for wa, ws, b, r
          in zip(wa[::decimation], ws[::decimation], bs[::decimation], ra[::decimation])]

speed = go.Scattergl(
    x=bs[::decimation] * np.sin(twa_rad),
    y=bs[::decimation] * np.cos(twa_rad),
    mode='markers',
    marker=dict(
        size=6,
        line=dict(width=1),
        color=ws[::decimation],
        colorscale='Portland',
        showscale=True,
        colorbar=go.ColorBar(
            title="Wind speed"
        ),
        opacity=0.5
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
layout.xaxis.showticklabels=False
layout.yaxis.showticklabels=False

fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='speed.html', auto_open=False)
