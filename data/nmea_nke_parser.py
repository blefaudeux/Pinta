import pynmea2 as nmea
import plotly.offline as py
import plotly.graph_objs as go

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
                print("Unknown field: {}".format(sample.identifier()[:-1]))
                skipped_fields[key] = 1
            pass

    except (nmea.ParseError, nmea.nmea.ChecksumError) as e:
        # Corrupted data, skip
        pass

# Reorganize for faster processing afterwards :
speed = [sample[4] for sample in data['Speed']]
awa = [sample[0] for sample in data['ApparentWind']]
# awa_r = [sample[0] for sample in data['ApparentWind'] and sample[1] == 'R']


# Display some data
# # - Linear traces
# boat_speed = go.Scatter(
#     y=[sample[4] for sample in data['Speed']],
#     mode='markers',
#     name='Boat Speed'
# )
#
# wind_speed = go.Scatter(
#     y=[sample[4] for sample in data['WindTrue']],
#     mode='markers',
#     name='Wind Speed'
# )
#
# traces = [boat_speed, wind_speed]
# py.plot(traces, filename='speed.html', auto_open=False)

# - polar plot
speed = go.Scatter(
    r=speed,
    t=awa,
    mode='markers',
    name='Boat speed',
    marker=dict(
        color='rgb(27,158,119)',
        size=110,
        line=dict(
            color='white'
        ),
        opacity=0.7
    )
)

traces = [speed]
layout = go.Layout(
    title='Speed relative to apparent wind direction',
    font=dict(
        size=15
    ),
    plot_bgcolor='rgb(223, 223, 223)',
    angularaxis=dict(
        tickcolor='rgb(253,253,253)'
    )
)
fig = go.Figure(data=traces, layout=layout)
py.plot(fig, filename='speed.html', auto_open=False)

