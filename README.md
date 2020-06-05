# Hands On

1. Available packages for calibration
2. Visualisation tools
    - Reliability diagram
    - Calibration maps
    - Expected Calibration Error
    - Maximum Calibration Error
3. Classifier ouptuts
    - Binary
    - Ternary
    - Multiclass
4. Non-neural demonstrations
    - Example of calibration
    - Results after calibration
5. Neural demonstrations
    - Output scores from a state-of-the-art Deep Neural Networks (DNNs)
    - Results after calibration of DNNs
6. The pipeline on how to train and evaluate classifiers and calibrators
    - Dataset partitioning for calibration
    - Calibration evaluation
    - Statistical tests for calibration
    - Statistical tests for model comparison

## Background

## Installation

Clone the code

```
git clone git@github.com:classifier-calibration/hands_on.git
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
pip install -r requirements.txt
```

## Repositories with notebooks related to calibration

This is a list of all material that I have created during last 4 years about
Calibration, some of these are unfinished projects and ideas and need to be
reorganised.

- [Beta calibration tutorial](https://github.com/betacal/python/blob/master/tutorial/Python%20tutorial.ipynb) (year 2017)
- [Dirichlet Calibration](https://github.com/dirichletcal/experiments_neurips/tree/master/notebooks) (year 2018)
    - [3-Class Calibration examples](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/Calibration_example.ipynb)
    - [Exploration of Dirichlet Cal implementations](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/Dirichlet_calibration.ipynb)
    - [Synthetic Gaussian Mixtures](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/synthetic_data_gaussian_mixture.ipynb)
    - [NeurIPS Softmax example v1](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/synthetic_data_gaussian_mixture_softmax_v_01.ipynb)
    - [NeurIPS Softmax example v2](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/synthetic_data_gaussian_mixture_softmax_v_02.ipynb)
    - [Multiple Toy examples](https://github.com/dirichletcal/experiments_neurips/blob/master/notebooks/toy_example_experiment.ipynb)
- [Bayesian Calibration](https://github.com/perellonieto/bayesian_calibration/tree/master/jupyter) (year 2016)
    - [Bernoulli Beta Prior](https://github.com/perellonieto/bayesian_calibration/blob/master/jupyter/Bernoulli_Beta_prior.ipynb)
    - [Gaussian Process Calibration](https://github.com/perellonieto/bayesian_calibration/blob/master/jupyter/GaussianProcess_calibration.ipynb)
- [Deep Calibration](https://github.com/perellonieto/deep_calibration) (year 2016)
    - [Density estimation ussing GMM](https://github.com/perellonieto/deep_calibration/blob/master/jupyter/Density_estimation_GMM.ipynb)
    - [Expectation Maximisation for GMM](https://github.com/perellonieto/deep_calibration/blob/master/jupyter/EM_Gaussian_mixture.ipynb)
- [Deep Beta Calibration](https://github.com/perellonieto/deep_betacal) (year 2018)
    - [visualisations.py](https://github.com/perellonieto/deep_betacal/blob/master/utils/visualisations.py)


## Slides and text material

- Beta Calibration
    - [Slides Aistats 2017](https://github.com/betacal/aistats2017/blob/master/aistats2017_beta_calibration_slides.pdf)
- Dirichlet Calibration
    - [Slides NeurIPS](https://dirichletcal.github.io/documents/neurips2019/slides.pdf)
    - [Poster NeurIPS](https://dirichletcal.github.io/documents/neurips2019/poster.pdf)
    - [8min PyData Bristol talk](https://docs.google.com/presentation/d/1RMzzNyQUz6BLQYCqD6RZT3ju__5fG4MbgNNmDkmRYDQ/edit#slide=id.g6b70f9ecd5_0_17)
    - [3min video slides](https://docs.google.com/presentation/d/1iQ-4hScB4WuonkSpKsXpRSvzTGLgT2LwFYvAeXHmI_o/edit#slide=id.g65639b587c_0_113)
