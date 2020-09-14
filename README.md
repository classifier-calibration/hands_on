# Hands On Classifier Calibration

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/classifier-calibration/hands_on/master)

## Table of contents

1. Hands On
2. Installation (optional)
3. Available packages for calibration
4. Additional material
5. Collaborate

## Hands On

The hands on is divided into the following sections

1. [Visualisation and evaluation tools](https://github.com/classifier-calibration/hands_on/blob/master/notebooks/1_visualisation_tools.ipynb)
2. [How to train a calibrator](https://github.com/classifier-calibration/hands_on/blob/master/notebooks/2_training_a_calibrator.ipynb)
3. Examples of classifiers and calibrators
    1. [Binary](https://github.com/classifier-calibration/hands_on/blob/master/notebooks/3_binary_examples.ipynb)
    2. [Ternary](https://github.com/classifier-calibration/hands_on/blob/master/notebooks/3_ternary_examples.ipynb)
4. [How to compare calibrator performance](https://github.com/classifier-calibration/hands_on/blob/master/notebooks/4_pipeline_train_evaluate.ipynb)
        
There are three options to follow the Hands On:

1. Interactive by runing the code in your computer (following the installation steps below).
2. Interactive in your webbrowser with Blinder by clicking the following button [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/classifier-calibration/hands_on/master) (no installation is required).
3. Static (pre-rendered) notebooks by clicking in each of the prevous sections (if it fails, try reloading).

## Installation (optional)

The Hands On can be run directly in your webbrowser with no installation (go to section Hands On). However, if you want to run the Hands On code in your own machine follow the next steps.

The following instructions has been tested in a Linux machine with Ubuntu 18.04 and Python 3.6. If you know how to install in other Operating systems feel free to add the information in a pull request. It is possible that the code does not work on Windows (the Dirichlet calibration library uses Jax which does not support natively Windows).

Clone this repository

```
git clone --recurse-submodules https://github.com/classifier-calibration/hands_on.git
cd hands_on
git submodule update
```

This code is working with Python3.6. 

```bash
# Use the module venv to create a virtual environment in a folder called venv
python3.6 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Upgrade package installer for Python
pip install --upgrade pip
# Install all the required dependencies indicated in the requirements.txt file
pip install -r binder/requirements.txt
pip install -e lib/PyCalib
pip install -e lib/dirichlet_python
# Create a kernel with the current virtual environment
python -m ipykernel install --user --name ClaCal --display-name "ClaCal handson"
```

Then you can start a Jupyter Notebook, and load the created kernel to run the
Hands On.

```
jupyter notebook
```

### Dockerfile (optional)
Running Notebooks on a Docker container.
```bash
git clone --recurse-submodules https://github.com/classifier-calibration/hands_on.git
cd hands_on
# build docker image from Dockerfile
export APPUSER=appuser
export IMGNAME_VER=clacal_hands_on:latest
docker build . -t $IMGNAME_VER
# run a container from new image
docker run --rm --name clacal -d -v "$PWD":/home/$APPUSER/data -p 8889:8889 $IMGNAME_VER
# Open browser to http://localhost:8889 (pass: clacal2020)
``` 

## Available packages for calibration

- Scikit-learn:
    - [Calibrated classifiers with cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
    - [Isotonic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression)
    - Platt's scaling
- [Beta Calibration](https://pypi.org/project/betacal/)
- [Dirichlet Calibration](https://pypi.org/project/dirichletcal/)
- [PyCaLib](https://github.com/perellonieto/PyCalib): Python calibration library used in this Hands On
- [NetCal](https://pypi.org/project/netcal/)
    - includes metrics, reliability diagram and calibration for Neural Nets
- [Pakdaman R package](https://github.com/pakdaman/calibration): Binary Classifier Calibration Models including BBQ, ENIR, and ELiTE


## Additional material

- Beta Calibration
    - [Slides Aistats 2017](https://github.com/betacal/aistats2017/blob/master/aistats2017_beta_calibration_slides.pdf)
- Dirichlet Calibration
    - [Slides NeurIPS](https://dirichletcal.github.io/documents/neurips2019/slides.pdf)
    - [Poster NeurIPS](https://dirichletcal.github.io/documents/neurips2019/poster.pdf)
    - [Slides PyData Bristol talk](https://docs.google.com/presentation/d/1RMzzNyQUz6BLQYCqD6RZT3ju__5fG4MbgNNmDkmRYDQ/edit#slide=id.g6b70f9ecd5_0_17)
    - [3min video slides](https://docs.google.com/presentation/d/1iQ-4hScB4WuonkSpKsXpRSvzTGLgT2LwFYvAeXHmI_o/edit#slide=id.g65639b587c_0_113)

## Collaborate

The previous code, examples, list of libraries, and additional material does not cover everything available online. If you know anything missing in any section feel free to [open an issue](https://github.com/classifier-calibration/hands_on/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) including your suggestion, or make a [pull request](https://github.com/classifier-calibration/hands_on/pulls?q=is%3Apr+is%3Aopen+sort%3Aupdated-desc) with the corresponding changes.
