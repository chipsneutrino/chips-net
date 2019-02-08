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

def parameter_network_train(parameter, file, outputFile, imageSize, valFrac, testFrac,
							norm, batchSize, learningRate, numEpochs, noHit, noTime):

	# Print configuration to stdout
	print("Parameter Training: Beginning training on parameter: {0} ...".format(parameter))
	print("File:{0} Output:{1}".format(file, outputFile)) 
	print("ValFrac:{0} TestFrac:{1}".format(valFrac, testFrac))
	print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(norm, batchSize, learningRate, numEpochs))
	print("ImageSize:{0} noHit:{1} noTime:{2}".format(imageSize, noHit, noTime))  

	# Load, shuffle and split the data into the different samples
	train_data, val_data, test_data = utils.load_txt_file(file, 0, valFrac, testFrac)	

	# Split train and validation samples into parameters_labels and images
	train_parameter, train_images 	= utils.parameter_images(train_data, norm, noHit, noTime, parameter, imageSize)
	val_parameter, val_images 		= utils.parameter_images(val_data, norm, noHit, noTime, parameter, imageSize)

	# Split the test data into labels, energies, parameters and images
	test_labels, test_energies, test_parameters, test_images = utils.labels_energies_parameters_images(test_data, norm, noHit, noTime, imageSize)

	# Configure the image input_shape for the network depending on the number of channels
	input_shape = (imageSize, imageSize, 2)
	if noHit or noTime:
		input_shape = (imageSize, imageSize, 1)

	# Get the specific Keras CVN parameter model
	cvn_model = models.cvn_parameter_model(parameter, input_shape, learningRate)

	# Fit the model with the training data and store the training "history"
	train_history = cvn_model.fit(train_images, train_parameter,
								batch_size=batchSize,
								epochs=numEpochs, verbose=1,
								validation_data=(val_images, val_parameter),
								callbacks=[utils.callback_checkpoint("output/parameter_model.ckpt")])

	# Save history to training output
	utils.save_regression_history(train_history, outputFile)

	# Evaluate the test sample on the trained model and save output to file
	test_output = cvn_model.predict(test_images, verbose=0)
	utils.save_regression_output(test_labels, test_energies, test_parameters, test_output, outputFile)

def parameter_network_evaluate():
	print("Parameter Evaluate: Beginning evaluation...")

def parameter_network_optimise():
	print("Parameter Optimise: Beginning optimisation...")

def main():
	args = utils.parse_args() # Get the command line arguments
	if args.train:
		parameter_network_train(int(args.parameter), args.inputFile, args.output, int(args.imageSize),
								float(args.valFrac), float(args.testFrac), args.norm, int(args.batchSize), 
								float(args.lRate), int(args.epochs), args.noHit, args.noTime)
	elif args.eval:
		parameter_network_evaluate()
	elif args.Optimise:
		parameter_network_optimise()
	else:
		print("Error: Something had gone wrong!")

if __name__=='__main__':
	main()

