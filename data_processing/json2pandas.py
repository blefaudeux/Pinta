import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

"""
    Parse linear Json logs, return a Pandas dataframe.
"""

LOG = logging.getLogger("Json2Pandas")

# Normalize fields post-parsing, align with other inputs
_PRESETS = {
    "Upwind": {"wind_speed": 21},  # "wind_angle": 55.0,
    "Reaching": {"wind_speed": 23},  # "wind_angle": 110.0,
    "Downwind": {"wind_speed": 20},  # "wind_angle": 135.0,
}


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
    "Boat.Port.FoilRake": "port_foil_rake",
    "Boat.RudderRake": "rudder_rake",
    "Sails": "sails",
}


def parse_raw_json(
    filepath: Path, reference_lookup: Optional[pd.DataFrame] = None, *args
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:

    # Load raw values
    raw_load = pd.read_json(filepath)

    # Match the column names with the normalized ones
    normalized_fields = [_FIELDS[f] for f in list(raw_load.columns)]
    raw_load.columns = normalized_fields

    # If we have a reference lookup, check whether we have some extra data for this file
    if reference_lookup is not None:
        try:
            match = reference_lookup[reference_lookup["run"] == filepath.stem]
            for k, v in zip(reference_lookup.columns, match.values[0]):
                try:
                    raw_load[_FIELDS[k]] = v
                except KeyError:
                    # skip this field on purpose
                    pass

            # Unroll the init point
            for k, v in _PRESETS[match["initPoint"].item()].items():
                raw_load[k] = v

        except IndexError:
            LOG.info(
                f"No extra data found in the reference lookup for the file {filepath.stem}"
            )

    return raw_load
