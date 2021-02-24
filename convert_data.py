#!/usr/bin/env python3

""" Convert the NMEA data logs into pandas-compatible jsons """

import argparse
import logging
import multiprocessing
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Optional, List

import json

from pinta.data_processing.csv2pandas import parse_csv
from pinta.data_processing.json2pandas import parse_raw_json
from pinta.data_processing.nmea2pandas import parse_nmea
from pinta.data_processing.utils import save_json

LOG = logging.getLogger("DataConversion")
SUPPORTED_FILES = [".json", ".csv", ".nmea"]


def process_file(
    filepath: Path,
    args: argparse.Namespace,
    lut: Optional[Dict[str, str]] = None,
    metadata: List[Dict[str, Any]] = None,
) -> None:
    df = {".nmea": parse_nmea, ".csv": parse_csv, ".json": parse_raw_json}[filepath.suffix](
        filepath, lut
    )  # type: ignore

    # Fill in the gaps in data
    # Drop the collumns which are completely empty
    df = df.bfill().ffill().dropna(axis=1)  # interpolate() ?
    save_json(df, Path(args.data_export_path) / Path(filepath.stem + ".json"))
    LOG.info(f"File {filepath.stem } processed")

    # Handle the optional metadata files
    # TODO: Find if this metadata file has a matching run number,
    # copy the extra data as needed in the pandas DF


def handle_directory(args: argparse.Namespace):
    LOG.info(f"Processing {args.data_ingestion_path}")

    # List all the raw files to be processed
    def get_file_list(ext: str):
        return list(Path(args.data_ingestion_path).glob("**/*" + ext))

    filelist = [f for ext in SUPPORTED_FILES for f in get_file_list(ext)]
    LOG.info(f"Found {len(filelist)} candidate files.")

    # Extract the optional metadata files
    metadatas: List[Any] = []
    if args.metadata_root != "":
        for meta_file in filter(lambda x: args.metadata_root in str(x), filelist):
            LOG.info(f"Loading metadata: {meta_file}")
            with open(meta_file.absolute(), "r") as fileio:
                metadatas.append(json.load(fileio))

        # Make sure that we don't load the metadata files
        filelist = list(set(filelist) - set(filter(lambda x: args.metadata_root in str(x), filelist)))

    # Make the export directory
    Path(args.data_export_path).mkdir(parents=True, exist_ok=True)

    # Get the optional lookup table
    # Expected fields are
    # [ 'run', 'ID', 'initPoint', 'Sails', 'Solver', 'Wind', 'Pilot', 'Ballasts', 'Ocean', 'TrameGen',
    #   'Boat.Port.RakeFoil', 'Boat.Port.FoilExtPerc', 'Boat.Aero.Sail.Flat',
    #   'Boat.KeelCant', 'Boat.CWATgt', 'BaseTWS_kts' ]
    lut = None
    if args.data_lookup_table != "":
        with open(args.data_lookup_table, "r") as fileio:
            lut = json.load(fileio)

        LOG.info("Provided look-up table: {}".format(lut))

    # Batch process all the files
    if args.parallel:
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        barrier = pool.starmap_async(process_file, zip(filelist, repeat(args), repeat(lut)))
        barrier.wait()
    else:
        for f in filelist:
            process_file(f, args, lut, metadatas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert in between data formats. Handles NMEA and some CSVs")
    parser.add_argument(
        "--data_ingestion_path",
        action="store",
        help="Full path of the root folder where the original data is",
    )

    parser.add_argument("--data_export_path", action="store", help="Where to save the normalized data")

    parser.add_argument(
        "--data_lookup_table",
        action="store",
        help="Path to an optional extra file which adds some context to all the files.",
        default="",
    )

    parser.add_argument(
        "--metadata_root",
        action="store",
        help="Root name of the metadata files",
        default="",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Convert muliple files in parallel. Errors may not be properly visible",
        default=False,
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
