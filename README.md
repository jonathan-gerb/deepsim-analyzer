# deepsim-analyzer
Image &amp; dataset analyzer based on deep similarity measures. Developed for the UvA Multimedia Analysis Course


# Development
To develop or make adjustments to the code, please install install the package locally in a new conda environment.

First create a new environment and install the requirements:
```
conda create -n ds-analyzer python=3.11
conda activate ds-analyzer
pip install -r requirements.txt
```
Next clone the repository and install it locally for development:
```
git clone https://github.com/jonathan-gerb/deepsim-analyzer
cd deepsim-analyzer
pip install -e .
```
You can start the GUI by calling.
```
ds-analyzer -i "path/to/images"
```
For a description of the parameters use the help argument:
```
ds-analyzer --help
```


# Using the GUI
## Installing
To use the GUI, please install first the package using pip:
```
pip install git+https://github.com/jonathan-gerb/deepsim-analyzer
```
now run the GUI with
```
ds-analyzer -i "path/to/images"
```