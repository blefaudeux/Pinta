#!/usr/bin/python3

""" Convert the NMEA data logs into pandas-compatible jsons """

import os
from data_processing.nmea2pandas import parse_nmea, save_json

WIND_BIAS = 5.


def handle_directory(path):
    # List the NMEA logs to process
    nmea_files = {}
    json_files = {}

    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.nmea'):
                filename = os.path.splitext(filename)[0]
                nmea_files[filename] = os.path.abspath(
                    os.sep.join([dirpath, filename])+'.nmea')

            if filename.endswith('.json'):
                json_files[os.path.splitext(filename)[0]] = 1

    # Convert the NMEA logs which are not already present as JSONs
    for nmea in nmea_files.keys():
        if nmea not in json_files.keys():
            print("Converting {}".format(nmea + '.nmea'))
            df = parse_nmea(nmea_files[nmea], WIND_BIAS)
            save_json(df, os.path.join("data/", nmea + '.json'))


if __name__ == '__main__':
    handle_directory("data")
