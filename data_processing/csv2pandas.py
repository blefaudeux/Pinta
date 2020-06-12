import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

"""
    Parse linear CSV logs, return a Pandas dataframe.
    Extra functions to load/save json files to speed up subsequent uses.
"""

LOG = logging.getLogger("CSV2Pandas")


def parse_csv(filepath: Path, wind_bias=0):
    raw_load = pd.read_csv(filepath)

    # TODO: @lefaudeux / match original and common downstream fields
    # Could be loaded on the fly via a json in the raw dump
    return raw_load
