"""Plotting module

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module automates the generation of output plots from a specific
experiment. This includes how the training progressed and the performance
tested on the beam test set. All plots are saved to a root file in 
the experiment directory.

Features
1) Generates loss and accuracy/mse plots with epoch number
2) Generates distribution histograms of the "labels"
3) Generates a performance plots depending on the model outputs
4) Runs over the beam "test" sets to see what the performance would be in the beam
5) Generates a t-sne plots by looking at the last dense layer on a test sample
6) Should be able to do this for every epoch so it can generate gifs
7) Takes in a config file that determines how events should be weighted and the ranges of plots etc...
8) Fits output plots with gaussians
9) Also should plot events that maximise a specific classification
10) Plots events that are classified wrong but have a strong value in another category
11) Plots a confusion matrix
12) I SHOULD DEVELOP ALL OF THIS IN A NOTEBOOK
"""

import ROOT