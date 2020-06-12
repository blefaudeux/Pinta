
import json
from pathlib import Path

import pandas as pd


def save_json(dataframe: pd.DataFrame, filepath: Path):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open("w") as file_out:
        json.dump(dataframe.to_json(), file_out)


def load_json(filepath: Path, skip_zeros=True) -> pd.DataFrame:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open("r") as file_in:
        data = json.load(file_in)

    dataframe = pd.read_json(data)
    return dataframe if not skip_zeros else dataframe[dataframe.boat_speed > 0].dropna()
