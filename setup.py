#!/usr/bin/env python3


import os
import re
import warnings

import setuptools
import torch

this_dir = os.path.dirname(os.path.abspath(__file__))


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


# https://packaging.python.org/guides/single-sourcing-package-version/
def find_version(version_file_path):
    with open(version_file_path) as version_file:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


extensions = []
cmdclass = {}


if __name__ == "__main__":
    setuptools.setup(
        name="pinta",
        description="experimenting ML and sailing",
        version=find_version("pinta/__init__.py"),
        install_requires=fetch_requirements(),
        include_package_data=True,
        packages=setuptools.find_packages(exclude=("tests", "tests.*")),
        ext_modules=extensions,
        cmdclass=cmdclass,
        python_requires=">=3.6",
        author="Benjamin Lefaudeux",
        author_email="banana@split.com",
        long_description="Read Joshua Slocum and save a whale",
        long_description_content_type="text/markdown",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "License :: OSI Approved :: GPL License",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Operating System :: OS Independent",
        ],
    )
