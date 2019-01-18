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
	parser.add_argument('-o', '--output',    default='output/pid.txt', help = 'Output .txt file')
	parser.add_argument('--train', action = 'store_true',       help = 'Use to train the network')
	parser.add_argument('--eval',  action = 'store_true',       help = 'Use to evaluate on the network')
	parser.add_argument('-v', '--valFrac',   default = 0.1,     help = 'Fraction of events for validation (0.1)')
	parser.add_argument('-t', '--testFrac',  default = 0.1,     help = 'Fraction of events for testing (0.1)')
	parser.add_argument('-b', '--batchSize', default = 500,     help = 'Training batch size (500)')
	parser.add_argument('-l', '--lRate',     default = 0.001,   help = 'Training learning rate (0.001)')
	parser.add_argument('-e', '--epochs',    default = 10,      help = 'Training epochs (10)')
	parser.add_argument('--noHit',  action = 'store_true', 		help = 'Do not use hit channel')
	parser.add_argument('--noTime', action = 'store_true', 		help = 'Do not use time channel')
	parser.add_argument('--norm',   action = 'store_true',		help = 'Normalise the channels')

	return parser.parse_args()	    

def pid_network_train(file, outputFile, valFrac, testFrac, norm,
					  batchSize, learningRate, numEpochs, noHit, noTime):

	# Print configuration to stdout
	print("Beginning Training -> ValidationFrac:{0} TestingFrac:{1}".format(
		  valFrac, testFrac))    
	print("Beginning Training -> Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
		  norm, batchSize, learningRate, numEpochs))  

	# Load, shuffle and split the data into the different samples
	train_data, val_data, test_data = utils.load_txt_file(file, 0, valFrac, testFrac)

	# Split train and validation samples into labels and normalised images
	train_labels, train_images = utils.labels_images(train_data, norm, noHit, noTime)
	val_labels, val_images = utils.labels_images(val_data, norm, noHit, noTime)
	test_labels, test_energies, test_images = utils.labels_energies_images(test_data, norm, noHit, noTime)

	# Configure the image input_shape for the network depending on the number of channels
	input_shape = (32, 32, 2)
	if noHit or noTime:
		input_shape = (32, 32, 1)

	# Get the keras CVN PID model
	categories = 5
	cvn_model = models.cvn_pid_model(categories, input_shape, learningRate)
   	
	# Fit the model with the training data and store the training "history"
	train_history = cvn_model.fit(train_images, train_labels,
								  batch_size=batchSize,
				  				  epochs=numEpochs, verbose=1,
				  				  validation_data=(val_images, val_labels),
				  				  callbacks=[utils.callback_checkpoint("output/pid_model.ckpt")])    

	score = cvn_model.evaluate(test_images, test_labels, verbose=0)     # Score the model
	print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(score[0], score[1]))

	# Save history to training output
	utils.save_category_history(train_history, outputFile)

	# Evaluate the test sample on the trained model and save output to file
	test_output = cvn_model.predict(test_images, verbose=0)
	utils.save_category_output(categories, test_labels, test_energies, test_output, outputFile)

def pid_network_evaluate():
	print("Evaluate: Beginning evaluation...")
	#cvn_model.load_weights(checkpoint_path)

def main():
	args = parse_args() # Get the command line arguments

	if args.noHit and args.noTime:
		print("Error: Need to use at least one channel")
		sys.exit()		

	if args.train and args.eval:
		print("Error: Can't train and evaluate at the same time")
		sys.exit()
	elif args.train and not args.eval:
		pid_network_train(args.file, args.output, float(args.valFrac), float(args.testFrac),
						  args.norm, int(args.batchSize), float(args.lRate), int(args.epochs),
						  args.noHit, args.noTime)
	elif args.eval and not args.train:
		pid_network_evaluate(args.file)
	else:
		print("Error: Need to specify --train or --eval")
		sys.exit()

if __name__=='__main__':
	main()
