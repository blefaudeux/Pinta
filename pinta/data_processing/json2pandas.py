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


def parse_raw_json(
    filepath: Path, data_lut: Optional[Dict] = None, *args
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:

    # Load raw values
    with open(filepath, "r") as fileio:
        loaded = simdjson.load(fileio)
        raw_load = (
            pd.DataFrame.from_dict(loaded)
            if isinstance(loaded, dict)
            else pd.read_json(loaded)
        )

    # If we have a reference lookup
    if data_lut is not None:
        # Delete the collumns which are explicitly marked null
        for key in data_lut.keys():
            if data_lut[key] is None and key in raw_load.keys():
                del raw_load[key]
                LOG.debug(f"Deleting collumn {key} - as requested")

        # Match the column names with the normalized ones
        try:
            LOG.debug(f"Raw columns: {raw_load.columns}")
            normalized_fields = [data_lut[f] for f in list(raw_load.columns)]
        except KeyError as e:
            LOG.error(
                f"KeyError {e}\n *** Please use a matching conversion table. Keys: {list(raw_load.columns)}\n"
            )
            raise RuntimeError

        raw_load.columns = normalized_fields

    return raw_load
