# Hands On Classifier Calibration

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/classifier-calibration/hands_on/master)

1. Installation (optional)
2. Hands On
3. Available packages for calibration
4. Additional material

## Installation (optional)

The Hands On can be run directly in your webbrowser with no installation (go to section Hands On). However, if you want to run the Hands On code in your own machine follow the next steps.

The following instructions has been tested in a Linux machine with Ubuntu 18.04 and Python 3.6. If you know how to install in other Operating systems feel free to add the information in a pull request. It is possible that the code does not work on Windows (the Dirichlet calibration library uses Jax which does not support natively Windows).

Clone this repository

```
git clone https://github.com/classifier-calibration/hands_on.git
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

## Hands On

The hands on contains the following sections

1. Visualisation and evaluation tools
2. How to train a calibrator
3. Examples of classifiers and calibrators
4. How to compare calibrator performance
        
In order to start the Hands On follow this link [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/classifier-calibration/hands_on/master)

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
