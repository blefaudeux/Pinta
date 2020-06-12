#!/usr/bin/python3

""" Convert the NMEA data logs into pandas-compatible jsons """

import argparse
import logging
from pathlib import Path

from data_processing.csv2pandas import parse_csv
from data_processing.nmea2pandas import parse_nmea
from data_processing.utils import save_json

WIND_BIAS = 5.0

LOG = logging.getLogger("DataConversion")


def handle_directory(args: argparse.Namespace):
    # List the NMEA logs to process
    raw_files = {}
    filetype = ""

    # List all the raw files to be processed
    filelist = list(Path(args.data_ingestion_path).glob("**/*.nmea")) + list(
        Path(args.data_ingestion_path).glob("**/*.csv")
    )

    LOG.info(f"Found {len(filelist)} candidate files")

    for filepath in filelist:
        if filepath.suffix != ".json":
            # Grep the file type, error out if there's a mix
            assert (
                not filetype or filetype == filepath.suffix
            ), "This only handles all CSVs or all NMEA"
            filetype = filepath.suffix

            raw_files[filepath.stem] = filepath.resolve()

    # List all the files already processed, skip processing in that case
    processed_files = [p.stem for p in Path(args.data_export_path).glob("**/*.json")]

    # Convert the logs which are not already present as internal JSONs
    for file_key, file_path in raw_files.items():
        if file_key not in processed_files:
            print(f"Converting {file_key}")

            df = {".nmea": parse_nmea, ".csv": parse_csv}[filetype](
                Path(file_path), WIND_BIAS
            )

            save_json(df, Path(args.data_export_path) / Path(file_key + ".json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Convert in between data formats. Handles NMEA and some CSVs"
    )
    parser.add_argument(
        "--data_ingestion_path",
        action="store",
        help="Full path of the root folder where the original data is",
    )

    parser.add_argument(
        "--data_export_path", action="store", help="Where to save the normalized data"
    )

    args = parser.parse_args()

    # Gracefully close if no arguments
    if not args.data_ingestion_path:
        parser.print_help()
        exit(-1)

    # Default to saving data where the inputs where
    if not args.data_export_path:
        args.data_export_path = args.data_ingestion_path

    handle_directory(args)
