#!/usr/bin/env python
"""Setup script for NeoJAMS."""

import codecs
import os
import sys

from setuptools import find_packages, setup

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    sys.exit()

version = "0.1.0"

if sys.version_info < (3, 12):
    print("ERROR: NeoJAMS requires Python 3.12 or newer")
    sys.exit(1)

with codecs.open("README.md", encoding="utf-8") as readme_file:
    README = readme_file.read()

with codecs.open("HISTORY.md", encoding="utf-8") as history_file:
    HISTORY = history_file.read()

setup(
    name="neojams",
    version=version,
    description="JAMS: A JSON Annotated Music Specification",
    author="NeoJAMS development team",
    author_email="",
    url="https://github.com/marl/jams",
    packages=find_packages(),
    package_data={
        "neojams": [
            "schemata/*.json",
            "schemata/namespaces/*.json",
            "schemata/namespaces/tag/*.json",
        ]
    },
    include_package_data=True,
    long_description=README + "\n\n" + HISTORY,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="audio music json",
    license="ISC",
    install_requires=[
        "numpy >= 1.20",
        "pandas >= 2.0.0",
        "jsonschema >= 4.0.0",
        "mir_eval >= 0.7",
        "sortedcontainers >= 2.4.0",
    ],
    python_requires=">=3.12",
)
