# Imports
import os.path
import os.path as path
import sys
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import optimizers

import cvn_models as models
import cvn_utilities as utils

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate/Train CHIPS CVN PID Network')
	parser.add_argument('file', help = 'Path to input "image" .txt file')
	parser.add_argument('-n', '--norm', action = 'store_true', help = 'Norm the channels')
	return parser.parse_args()	  

def main():
	args = parse_args() # Get the command line arguments

	train_data, val_data, test_data = utils.load_txt_file(args.file, 0, 0.0, 0.0)	
	hitLabels, hitImages = utils.labels_images(train_data, args.norm, False, True)
	timeLabels, timeImages = utils.labels_images(train_data, args.norm, True, False)

	for event in range(100):
		utils.plot_image(hitImages, int(event))
		utils.plot_image(timeImages, int(event))

if __name__=='__main__':
	main()
