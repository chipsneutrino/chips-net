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
	parser.add_argument('-o', '--output',    default='out.txt', help = 'Output .txt file')
	parser.add_argument('--train', action = 'store_true',       help = 'Use to train the network')
	parser.add_argument('--eval',  action = 'store_true',       help = 'Use to evaluate on the network')
	parser.add_argument('-v', '--valFrac',   default = 0.1,     help = 'Fraction of events for validation (0.1)')
	parser.add_argument('-t', '--testFrac',  default = 0.1,     help = 'Fraction of events for testing (0.1)')
	parser.add_argument('-b', '--batchSize', default = 500,     help = 'Training batch size (500)')
	parser.add_argument('-l', '--lRate',     default = 0.001,   help = 'Training learning rate (0.001)')
	parser.add_argument('-e', '--epochs',    default = 10,      help = 'Training epochs (10)')
	parser.add_argument('--noHit',  action = 'store_true', 		help = 'Do not use hit channel')
	parser.add_argument('--noTime', action = 'store_true', 		help = 'Do not use time channel')

	return parser.parse_args()	    

def network_train(file, valFrac, testFrac, batchSize, lRate, epochs, outputFile, noHit, noTime):
	print("Train: Beginning training...")

	# Load, shuffle and split the data into the different samples
	train_data, val_data, test_data = utils.load_txt_file(file, 0, valFrac, testFrac)

	# Split train and validation samples into labels and normalised images
	train_labels, train_images = utils.labels_images(train_data, True, noHit, noTime)
	val_labels, val_images = utils.labels_images(val_data, True, noHit, noTime)

	utils.shift_labels_down(train_labels)
	utils.shift_labels_down(val_labels)

	input_shape = (32, 32, 2)
	if noHit or noTime:
		input_shape = (32, 32, 1)

	categories = 5
	cnn_model = models.cnn_model(categories, input_shape, lRate)       	# Get the Keras CNN model
   	
	history = cnn_model.fit(train_images, train_labels,					# Fit the model
							batch_size=int(batchSize),
				  			epochs=int(epochs), verbose=1,
				  			validation_data=(val_images, val_labels),
				  			callbacks=[utils.callback_checkpoint("model.ckpt")])    

	utils.plot_history(history)

	# Split the testing sample into labels, beam energies and normalised images
	test_labels, test_energies, test_images = utils.labels_energies_images(test_data, True, noHit, noTime)
	utils.shift_labels_down(test_labels)

	score = cnn_model.evaluate(test_images, test_labels, verbose=0)     # Score the model
	print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(score[0], score[1]))

	output = cnn_model.predict(test_images, verbose=0)                  # Evaluate the test sample

	# Save the test sample predictions to file for further analysis
	output_file = open(outputFile, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		for category in range(categories):
			out += (" " + str(output[num][category]))
		out += "\n"
		output_file.write(out)
	output_file.close()     

def network_evaluate():
	print("Evaluate: Beginning evaluation...")
	#cnn_model.load_weights(checkpoint_path)

def main():
	args = parse_args() # Get the command line arguments

	if args.noHit and args.noTime:
		print("Error: Need to use at least one channel")
		sys.exit()		

	if args.train and args.eval:
		print("Error: Can't train and evaluate at the same time")
		sys.exit()
	elif args.train and not args.eval:
		network_train(args.file, args.valFrac, args.testFrac, 
					  args.batchSize, args.lRate, args.epochs,
					  args.output, args.noHit, args.noTime)
	elif args.eval and not args.train:
		network_evaluate(args.file)
	else:
		print("Error: Need to specify --train or --eval")
		sys.exit()

if __name__=='__main__':
	main()
