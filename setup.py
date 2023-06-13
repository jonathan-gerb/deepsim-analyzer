"""Setup the package."""
from setuptools import setup, find_packages

# Version number

import re
VERSIONFILE = "deepsim_analyzer/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")


with open("README.md", "r") as fh:
    long_description = fh.read()

# Alle requirements
requirements = []

setup(
    name="deepsim_analyzer",
    version=verstr,
    author="Jonathan Gerbscheid",
    author_email="jonathan.gerbscheid@protonmail.com",
    description="Image & dataset analyzer based on many different similarity measures. Developed for the UvA Multimedia Analysis course",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonathan-gerb/deepsim-analyzer",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: All rights reserved",
        "Operating System :: OS Independent",
        'Development Status :: 3 - Alpha'
    ],
    entry_points={
        "console_scripts" : ['ds-analyzer=deepsim_analyzer:main']
    },
    python_requires=">=3.9",
    # zip_safe=False
)
