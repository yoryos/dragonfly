# Dragonfly Neural Simulation
A simulation of the neural mechanisms for dragonfly target tracking.

## Installation
To install the dragonfly simulation and get python dependancies run:
`./install.sh`

For this to work ensure the following dependacies are installed:

`python-pip,python-imaging, libjpeg62, libjpeg62-dev, libcairo2-dev, libffi-dev, python-opencv, python-qt4-gl, python-matplotlib`

The code was tested primarly on Ubuntu 14.04.4 LTS and the CSTMD1 simulation requires a CUDA enabled nVidia GPU.

## Running Instructions

### Data Requirement

There are some data requirements needed to run this project. Example data can be downloaded from:
https://www.doc.ic.ac.uk/~cps15/Dragonfly/Example_Data/ together with a guide on how to load it to use the various modules.

### DragonflyBrain

WARNING: Morphology data is required to run this module. Download this from the example data url given and place in a directory. The path of this directory should be specified as the config file parameter morphology_path. e.g morphology_path = Integration/CSTMD1_Data

```python
python -m Integration.run
```

### STDP module

To run the STDP module, the runStdp.py is called with a command-line argument which specifies the section in the runStdpConfig.ini file that holds the desired parameters
and folder names for test-samples, weights and directories for saving output-data

```python
    python -m STDP.runStdp SECTION
```

### Visualisers

To run the dragonfly visualisers, paths to the output data must be set in Visualisers/run.py
Then use:

```python
python -m Visualisers.run
```

from the root directory


### Tests

To run the complete test suite use:

```python
python test_dragonfly_suite.py
```

Do not attempt to use python -m unittest ... as a QtGui instance must be managed to run the visualiser tests


### CSTMD1

WARNING: Morphology data is required to run this module. Download this from the example data url given and place in a directory. The path of this directory should be specified as the config file parameter morphology_path. e.g morphology_path = Integration/CSTMD1_Data

### ESTMD1

To run the live estmd demonstration use:

```python
python -m ESTMD.live_estmd
```

### Environment

An example of script showing how to call the Environment is given in as Environment/example_environment.py
