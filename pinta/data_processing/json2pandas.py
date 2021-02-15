import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import simdjson

"""
    Parse linear Json logs, return a Pandas dataframe.
"""

LOG = logging.getLogger("Json2Pandas")

# Normalize fields post-parsing, align with other inputs
_PRESETS = {
    "Upwind": {"tws": 21},  # "twa": 55.0,
    "Reaching": {"tws": 23},  # "twa": 110.0,
    "Downwind": {"tws": 20},  # "twa": 135.0,
}


_FIELDS = {
    "Block.Time": "time",
    "Block.Boat.Speed_kts": "sog",
    "Block.Boat.Leeway": "leeway",
    "Block.Boat.Trim": "trim",
    "Block.Boat.Heel": "heel",
    "Block.Boat.TWA": "twa",
    "Block.Boat.Helm": "helm",
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
    filepath: Path, data_lut: Optional[Dict] = None, *args
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:

    # Load raw values
    with open(filepath, "r") as fileio:
        json = simdjson.load(fileio)
        raw_load = pd.read_json(json)

    # If we have a reference lookup
    if data_lut is not None:
        # Delete the collumns which are explicitly marked null
        for key in data_lut.keys():
            if data_lut[key] is None:
                LOG.debug(f"Deleting collumn {key} - as requested")
                del raw_load[key]

        # Match the column names with the normalized ones
        try:
            normalized_fields = [data_lut[f] for f in list(raw_load.columns)]
        except KeyError as e:
            LOG.error(f"KeyError {e}\n *** Please use a matching conversion table. Keys: {list(raw_load.columns)}\n")

        raw_load.columns = normalized_fields

        # #  check whether we have some extra data for this file
        # try:
        #     match = reference_lookup[reference_lookup["run"] == filepath.stem]
        #     for k, v in zip(reference_lookup.keys(), match.values[0]):
        #         try:
        #             raw_load[_FIELDS[k]] = v
        #         except KeyError:
        #             # skip this field on purpose
        #             pass

        #     # Unroll the init point
        #     for k, v in _PRESETS[match["initPoint"].item()].items():
        #         raw_load[k] = v

        # except IndexError:
        #     LOG.info(f"No extra data found in the reference lookup for the file {filepath.stem}")

    return raw_load
