import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pynmea2 as nmea

"""
    Parse linear NMEA logs, return a Pandas dataframe.
    Extra functions to load/save json files to speed up subsequent uses.
"""

LOG = logging.getLogger("NMEA2Pandas")

# The fields that we know of, and which can be parsed into the dataframe
nmea_fields = {
    "IIVHW": "Speed",
    "IIVLW": "Log",
    "IIDPT": "DepthSetup",
    "IIDBT": "Depth",
    "IIMTW": "WaterTemp",
    "IIVWR": "ApparentWind",
    "IIMWD": "WindTrue",
    "IIVWT": "WindTrueLR",
    "IIMTA": "AirTemp",
    "IIHDG": "HeadingMag",
    "IIHDM": "HeadingMagPrecise",
    "IIHDT": "HeadingTrue",
    "IIZDA": "Time",
    "IIGLL": "Pose",
    "IIVGT": "BottomHeading",
    "IIXTE": "CrossTrackError",
    "IIRSA": "RudderAngle",
}


WIND_BIAS = 5.0


def parse_nmea(filepath: Path, wind_bias=WIND_BIAS, *args):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open("r") as fileIO:
        data: Dict[str, Any] = {v: {} for _, v in nmea_fields.items()}

        # Parse everything & dispatch in the appropriate categories
        skipped_fields: Dict[str, int] = {}
        timestamp = None
        LOG.info("Parsing the NMEA file {}".format(filepath))
        LOG.info("Wind direction bias: {}".format(wind_bias))
        for line in fileIO:
            if line == "\n":
                continue

            try:
                sample = nmea.parse(line)
                key = sample.identifier()[:-1]

                if key == "IIZDA":
                    timestamp = time.mktime(sample.datetime.timetuple())

                try:
                    if timestamp is not None:
                        # One measurement per timestamp, for now
                        data[nmea_fields[key]][timestamp] = sample.data

                except KeyError:
                    # We discard this field for now
                    if key not in skipped_fields.keys():
                        LOG.warning(
                            "Unknown field: {}".format(sample.identifier()[:-1])
                        )
                        skipped_fields[key] = 1

            except (nmea.ParseError, nmea.nmea.ChecksumError, TypeError) as exception:
                # Corrupted data, skip
                LOG.warning(exception)

    # Reorganize for faster processing afterwards, save in a pandas dataframe
    LOG.info("Reorganizing data per timestamp")
    wa = []
    wa_index = []
    for ts in data["Speed"].keys():
        if ts in data["WindTrue"].keys():
            try:
                wa.append(
                    (
                        float(data["Speed"][ts][0])
                        - float(data["WindTrue"][ts][0])
                        + wind_bias
                        + 180
                    )
                    % 360.0
                    - 180.0
                )
                wa_index.append(ts)

            except ValueError:
                # Corrupted data, skip
                pass

    dataframe = {
        "twa": pd.Series(wa, index=wa_index),
        "sog": pd.Series(
            [float(data["Speed"][ts][4]) for ts in data["Speed"].keys()],
            index=data["Speed"].keys(),
        ),
        "tws": pd.Series(
            [float(data["WindTrue"][ts][4]) for ts in data["WindTrue"].keys()],
            index=data["WindTrue"].keys(),
        ),
        "helm": pd.Series(
            [float(data["RudderAngle"][ts][0]) for ts in data["RudderAngle"].keys()],
            index=data["RudderAngle"].keys(),
        ),
    }

    return pd.DataFrame(dataframe)
