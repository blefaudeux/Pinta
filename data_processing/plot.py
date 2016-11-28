import plotly.graph_objs as go
import plotly.offline as py
import numpy as np


# Display some data
def speed_plot(df, decimation=2):
    print("\nPlotting data")

    # - raw polar plot
    twa_rad = np.radians(df['wind_angle'][::decimation])
    labels = ['Wind: {}deg - {}kt \nBoat speed: {}kt * Rudder: {}deg'.format(wa, ws, b, r) for wa, ws, b, r
              in zip(df['wind_angle'][::decimation],
                     df['wind_speed'][::decimation],
                     df['boat_speed'][::decimation],
                     df['rudder_angle'][::decimation])]

    speed = go.Scattergl(
        x=df['boat_speed'][::decimation] * np.sin(twa_rad),
        y=df['boat_speed'][::decimation] * np.cos(twa_rad),
        mode='markers',
        marker=dict(
            size=6,
            color=df['wind_speed'][::decimation],
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
        title='Speed vs True Wind',
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


def rudder_plot(df):
    data = [
        go.Histogram(
            x=df['rudder_angle']
        )
    ]
    py.plot(data, filename='rudder_histogram.html', auto_open=False)