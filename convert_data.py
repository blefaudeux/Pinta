#!/usr/bin/python3

""" Convert the NMEA data logs into pandas-compatible jsons """

import argparse
import logging
import multiprocessing
from itertools import repeat
from pathlib import Path

from data_processing.csv2pandas import parse_csv
from data_processing.json2pandas import parse_raw_json
from data_processing.nmea2pandas import parse_nmea
from data_processing.utils import save_json

WIND_BIAS = 5.0

LOG = logging.getLogger("DataConversion")


def process_file(filepath: Path, args: argparse.Namespace) -> None:
    df = {".nmea": parse_nmea, ".csv": parse_csv, ".json": parse_raw_json}[
        filepath.suffix
    ](filepath, WIND_BIAS)

    save_json(df, Path(args.data_export_path) / Path(filepath.stem + ".json"))
    LOG.info(f"File {filepath.stem } processed")


def handle_directory(args: argparse.Namespace):
    LOG.info(f"Processing {args.data_ingestion_path}")

    # List all the raw files to be processed
    def get_file_list(ext: str):
        return list(Path(args.data_ingestion_path).glob("**/*" + ext))

    filelist = get_file_list(".nmea") + get_file_list(".csv") + get_file_list(".json")
    LOG.info(f"Found {len(filelist)} candidate files")

    # Make the export directory
    Path(args.data_export_path).mkdir(parents=True, exist_ok=True)

    # Batch process all the files
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.starmap_async(process_file, zip(filelist, repeat(args)))
    pool.close()
    pool.join()


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
    if not args.data_ingestion_path or not args.data_export_path:
        parser.print_help()
        exit(-1)

    # Default to saving data where the inputs where
    if args.data_ingestion_path == args.data_export_path:
        logging.error("Input and destination folders cannot be the same")
        exit(-1)

    logging.basicConfig(level=logging.INFO)
    handle_directory(args)
