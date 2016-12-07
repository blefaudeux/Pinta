import plotly.graph_objs as go
import plotly.offline as py
import numpy as np

""" Several helper functions to produce plots, pretty self-explanatory """


# Speed polar plot
def speed_plot(df, decimation=2, filename='speed_polar'):
    print("\nPlotting data")

    # - raw polar plot
    twa_rad = np.radians(df['wind_angle'][::decimation])
    labels = ['Wind: {}deg - {}kt ** Boat: {}kt ** Rudder: {}deg'.format(wa, ws, b, r) for wa, ws, b, r
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

    r_x = [rr * np.cos(i / 180. * np.pi) for rr in range(0, 20, 2) for i in range(0, 361)]
    r_y = [rr * np.sin(i / 180. * np.pi) for rr in range(0, 20, 2) for i in range(0, 361)]
    circle_labels = [str(i) + ' knots' for i in range(0, 20, 2) for ii in range(0, 361)]

    # Create a trace
    iso_speed = go.Scattergl(
        x=r_x,
        y=r_y,
        mode='lines',
        text=circle_labels,
        line=dict(
            width=1,
            color='grey')

    )

    traces = [speed, iso_speed]

    layout = go.Layout(
        title='Speed vs True Wind',
        orientation=90,
        autosize=False,
        width=800,
        height=800,
        hovermode='closest',
        plot_bgcolor='rgb(245, 245, 245)'
    )
    layout.xaxis.showticklabels=False
    layout.yaxis.showticklabels=False
    layout.xaxis.showgrid = False
    layout.yaxis.showgrid = False
    layout.xaxis.range = [-13, 13]
    layout.yaxis.range = [-13, 13]

    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig, filename=filename+'.html', auto_open=False)


# Plot any traces in parallel
def parrallel_plot(data_list, legend_list, title=None):
    traces = []

    for data, name in zip(data_list, legend_list):
        traces.append(
            go.Scatter(
                y=data,
                name=name
                )
        )

    layout = go.Layout(
        title=title if title is not None else "",
        hovermode='closest'
    )

    fig = go.Figure(data=traces, layout=layout)
    py.plot(fig, filename=title+'.html', auto_open=False)


def rudder_plot(df, filename='rudder_histogram'):
    traces = [
        go.Histogram(
            x=df['rudder_angle']
        )
    ]

    layout = go.Layout(
        title='Rudder angle distribution',
        hovermode='closest'
    )

    fig = go.Figure(data=traces, layout=layout)

    py.plot(fig, filename=filename+'.html', auto_open=False)


def scatter_plot(data, axes, title=None):
    trace = go.Scattergl(
        x=data[0],
        y=data[1],
        mode='markers'
    )

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis=dict(
            title=axes[0]
        ),
        yaxis=dict(
            title=axes[1]
        )
    )

    py.plot(go.Figure(data=[trace], layout=layout), filename=title+'.html', auto_open=False)
