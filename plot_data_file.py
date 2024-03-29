#!/usr/bin/env python3

import argparse
from pathlib import Path

from pinta.data_processing.load import load
from pinta.data_processing.plot import multi_plot, speed_plot


def run(args):
    # Load the full json and do some analysis
    df = load(Path(args.filepath).absolute())

    # Purely sequential plot
    multi_plot(
        df,
        ["heel", "tws", "twa", "sog"],
        "test multi plot",
        "multi_plot",
        True,
    )

    # Polar plot
    speed_plot(df, decimation=2, filename="speed_polar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize the contents of a data file")

    parser.add_argument(
        "filepath",
        action="store",
        help="Filepath of the .json file you want to visualize",
    )

    args = parser.parse_args()
    print(args.filepath)
    run(args)
