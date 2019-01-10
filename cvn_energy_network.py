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
	parser = argparse.ArgumentParser(description='Evaluate/Train CHIPS CVN Energy Estimation Network')

	parser.add_argument('file', help = 'Path to input "image" .txt file')
	parser.add_argument('-o', '--output',    default='out.txt', help = 'Output .txt file')
	parser.add_argument('--train', action = 'store_true',       help = 'Use to train the network')
	parser.add_argument('--eval',  action = 'store_true',       help = 'Use to evaluate on the network')
	parser.add_argument('-v', '--valFrac',   default = 0.1,     help = 'Fraction of events for validation (0.1)')
	parser.add_argument('-t', '--testFrac',  default = 0.1,     help = 'Fraction of events for testing (0.1)')
	parser.add_argument('-b', '--batchSize', default = 500,     help = 'Training batch size (500)')
	parser.add_argument('-l', '--lRate',     default = 0.001,   help = 'Training learning rate (0.001)')
	parser.add_argument('-e', '--epochs',    default = 10,      help = 'Training epochs (10)')

	return parser.parse_args()	    

def network_train(file, valFrac, testFrac, batchSize, lRate, epochs, outputFile):
	print("Train: Beginning training...")   

def network_evaluate():
	print("Evaluate: Beginning evaluation...")
	#cnn_model.load_weights(checkpoint_path)

def main():
	args = parse_args() # Get the command line arguments

	if args.train and args.eval:
		print("Error: Can't train and evaluate at the same time")
		sys.exit()
	elif args.train and not args.eval:
		network_train(args.file, args.valFrac, args.testFrac, 
					  args.batchSize, args.lRate, args.epochs,
					  args.output)
	elif args.eval and not args.train:
		network_evaluate(args.file)
	else:
		print("Error: Need to specify --train or --eval")
		sys.exit()

if __name__=='__main__':
	main()
