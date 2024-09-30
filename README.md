# Identifiability for Gaussian Processes with Stationary Holomorphic Kernels

This repository houses the complete codebase for the research study titled “Identifiability for Gaussian Processes with
Stationary Holomorphic Kernels”. 

## Repository Structure

This repository contains the code necessary for running the simulations in our study.

- **4.1**

  - `eps=0.1/rbf-new.py` - Python script for generating the results in Figure 1(a)
  - `eps=0.1/damped-periodic-new.py` - Python script for generating the results in Figure 1(b)
  - `eps=0.1/periodic-new.py` - Python script for generating the results in Figure 1(c)
  - `eps=0.1/rational-quadratic-new.py` - Python script for generating the results in Figure 1(d)
  - `eps=0.1/cosine-new.py` - Python script for generating the results in Figure 1(e)
  - `eps=0.1/setup.ipynb` - Jupyter notebook for setting up output directories
  - `eps=0.1/generate-boxplots.ipynb` - Jupyter notebook for generating the plots 

- **4.2**
  - `gp-for-ml.ipynb` - Python script for generating Figure 2

- **Additional Figures**
    Figure 3 uses all the files used to generate figure 1, and in addition uses the following:
  - `eps=0.01/rbf-new.py` - Python script for generating the results in Figure 1(a)
  - `eps=0.01/damped-periodic-new.py` - Python script for generating the results in Figure 1(b)
  - `eps=0.01/periodic-new.py` - Python script for generating the results in Figure 1(c)
  - `eps=0.01/rational-quadratic-new.py` - Python script for generating the results in Figure 1(d)
  - `eps=0.01/cosine-new.py` - Python script for generating the results in Figure 1(e)
  - `eps=0.01/setup.ipynb` - Jupyter notebook for setting up output directories
  - `eps=0.01/generate-boxplots.ipynb` - Jupyter notebook for generating the plots 

## Python Dependencies

Ensure your environment is set up with the following packages:

- NumPy version: 1.20.3
- SciPy version: 1.7.1
- Matplotlib version: 3.4.3
- Scikit-learn version: 0.24.2

## Instructions to run
Change directories to `eps=0.1`. Run `setup.ipynb`. Run each of the 5 .py scripts. Run `generate-boxplots.ipynb`.

Change directories back to the starting directory. Run `setup.ipynb` and `gp-for-ml.ipynb`.

Change directories to `eps=0.01`. Run `setup.ipynb`. Run each of the 5 .py scripts. Run `generate-boxplots.ipynb`.

All raw computed MLEs are stored in .npy files in the respective `results` directories. Boxplots generated from these .npy files are stored in the respective `boxplots` directories.

## Estimated Runtime
- **4.1**

  - `eps=0.1/rbf-new.py` - 8 hours, 53 minutes
  - `eps=0.1/damped-periodic-new.py` - 6 hours, 24 minutes
  - `eps=0.1/periodic-new.py` - 10 hours, 45 minutes
  - `eps=0.1/rational-quadratic-new.py` - 10 hours, 16 minutes
  - `eps=0.1/cosine-new.py` - 7 hours, 49 minutes
  - `eps=0.1/setup.ipynb` - 10 seconds
  - `eps=0.1/generate-boxplots.ipynb` - 10 seconds 

- **4.2**
  - `gp-for-ml.ipynb` - 1 hour, 21 minutes

- **Additional Figures**
  - `eps=0.01/rbf-new.py` - 7 hours, 30 minutes
  - `eps=0.01/damped-periodic-new.py` - 6 hours, 45 minutes
  - `eps=0.01/periodic-new.py` - 10 hours, 12 minutes
  - `eps=0.01/rational-quadratic-new.py` - 9 hours, 42 minutes
  - `eps=0.01/cosine-new.py` - 5 hours, 44 minutes
  - `eps=0.01/setup.ipynb` - 10 seconds
  - `eps=0.01/generate-boxplots.ipynb` - 10 seconds