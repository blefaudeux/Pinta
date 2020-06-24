import logging
from pathlib import Path

import pandas as pd

"""
    Parse linear Json logs, return a Pandas dataframe.
"""

LOG = logging.getLogger("Json2Pandas")

# Normalize fields post-parsing, align with other inputs
_FIELDS = {
    "Block.Time": "time",
    "Block.Boat.Speed_kts": "boat_speed",
    "Block.Boat.Leeway": "leeway",
    "Block.Boat.Trim": "trim",
    "Block.Boat.Heel": "heel",
    "Block.Boat.TWA": "wind_angle",
    "Block.Boat.Helm": "rudder_angle",
    "Block.Boat.Center.HullDisplPercent": "center_hull_disp",
    "Block.Boat.Center.BoardDisplPercent": "center_board_disp",
    "Block.Boat.Center.RudDisplPercent": "center_rudder_disp",
    "Block.Boat.Port.FoilAltitude": "port_foil_altitude",
    "Block.Boat.Port.FoilDisplPercent": "port_foil_disp",
    "Block.Boat.Port.HullDisplPercent": "port_hull_disp",
    "Block.Boat.Port.RudDisplPercent": "port_rud_disp",
    "Block.Boat.Stbd.FoilAltitude": "stbd_foil_altitude",
    "Block.Boat.Stbd.FoilDisplPercent": "stbd_foil_disp",
    "Block.Boat.Stbd.HullDisplPercent": "stbd_hull_disp",
    "Block.Boat.Stbd.RudDisplPercent": "stbd_rud_disp",
}


def parse_raw_json(filepath: Path, *args):
    # Load raw values
    raw_load = pd.read_json(filepath)

    # Match the column names with the normalized ones
    normalized_fields = [_FIELDS[f] for f in list(raw_load.columns)]
    raw_load.columns = normalized_fields

    return raw_load
