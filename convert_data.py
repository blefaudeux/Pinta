#!/usr/bin/env python3

""" Convert the NMEA data logs into pandas-compatible jsons """

import argparse
import logging
import multiprocessing
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd
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

    # Handle the optional metadata files
    if metadata is not None:
        # - find the run ID
        id_run, id_meta = None, -1
        for _ in metadata:
            id_meta += 1
            for k, v in metadata[id_meta]["run"].items():
                if v == filepath.stem:
                    id_run = k
                    break

        if id_run is not None:
            # - add the metadata in the DataFrame (take the lookup into account)
            LOG.info(f"Found ID for file {filepath.stem} : {id_run} in metadata file {id_meta}")

            for k in metadata[id_meta].keys():
                if k == "run" or k == "ID":
                    continue

                key = lut[k] if lut and k in lut.keys() else k
                if lut and k not in lut.keys():
                    LOG.debug(f"{k} key is missing from the look-up table")

                df[key] = metadata[id_meta][k][id_run]
                LOG.debug(f"{k} brings value: {metadata[id_meta][k][id_run]}")
        else:
            LOG.warning(f"Could not find the metadata for {filepath}")

    # Fill in the gaps in data
    # Drop the collumns which are completely empty
    df = df.bfill().ffill().dropna(axis=1)  # interpolate() ?

    if args.pickle:
        pd.to_pickle(df, Path(args.data_export_path) / Path(filepath.stem + ".pkl"), protocol=4)
        # test = pd.read_pickle(Path(args.data_export_path) / Path(filepath.stem + ".pkl"))
        # print(df.columns)
        # print(test.columns)
        # exit(-1)
    else:
        save_json(df, Path(args.data_export_path) / Path(filepath.stem + ".json"))

    LOG.info(f"File {filepath.stem } processed")


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
        LOG.info(f"{len(filelist)} files to process, excluding metadata.")

    # Make the export directory
    Path(args.data_export_path).mkdir(parents=True, exist_ok=True)

    # Get the optional lookup table
    lut = None
    if args.data_lookup_table != "":
        with open(args.data_lookup_table, "r") as fileio:
            lut = json.load(fileio)

        LOG.info("Provided look-up table: {}".format(lut))

    # Batch process all the files
    if args.parallel:
        LOG.info("Starting parallel conversion")
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        barrier = pool.starmap_async(process_file, zip(filelist, repeat(args), repeat(lut), repeat(metadatas)))
        barrier.wait()
    else:
        LOG.info("Starting sequential conversion")
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

    parser.add_argument(
        "--pickle",
        action="store_true",
        help="Save data files as pickled files -more compact, faster to load-",
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
