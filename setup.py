#!/usr/bin/env python

"""The setup script."""

import pathlib

from setuptools import find_packages, setup

def read_requirements(file_path):
    with open(file_path, "r") as f:
        requirements = f.read().splitlines()
    return requirements

requirements = read_requirements("requirements.txt")
setup_requirements = requirements.copy()
test_requirements = []

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    author="Jesus Gonzalez Ferrer",
    author_email="jgonz373@ucsc.edu",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Foundational SIMS",
    install_requires=requirements.copy(),
    license="MIT license",
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="simsational",
    name="simsational",
    packages=find_packages(exclude=["tests"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/JesusGF1/SIMSational",
    version="1.0.0",
    zip_safe=False,
)

