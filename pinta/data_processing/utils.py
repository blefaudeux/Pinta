from pathlib import Path

import pandas as pd
import simdjson
from io import StringIO


def save_json(dataframe: pd.DataFrame, filepath: Path):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open("w") as file_out:
        simdjson.dump(dataframe.to_json(), file_out)


def load_json(filepath: Path, skip_zeros=True) -> pd.DataFrame:
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with filepath.open("r") as file_in:
        dataframe = pd.read_json(StringIO(simdjson.load(file_in)))  # type: ignore

    try:
        return dataframe if not skip_zeros else dataframe[dataframe.sog > 0].dropna()
    except AttributeError:
        return dataframe.dropna()
