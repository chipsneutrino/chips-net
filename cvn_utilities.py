import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt
import os
import argparse

###################################################
#            Network Argument Handling            #
###################################################

# Parse the command line argument options
def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate/Train CHIPS CVN PID Network')

	# Input and output files (history file will have similar name to output file)
	parser.add_argument('inputFile', help = 'Path to combined input "image" .txt file')
	parser.add_argument('-o', '--output',    default='output/output.txt', help = 'Output .txt file')

	# What function are we doing?
	parser.add_argument('--train', 		action = 'store_true',  help = 'Use to train the network')
	parser.add_argument('--eval',  		action = 'store_true',  help = 'Use to evaluate on the network')
	parser.add_argument('--optimise', 	action = 'store_true', 	help = 'Use to optimise the network')

	# Network Hyperparameters
	parser.add_argument('-p', '--parameter', default = 6,		help = 'Parameter to fit (lepton Energy = 6)')
	parser.add_argument('-v', '--valFrac',   default = 0.1,     help = 'Fraction of events for validation (0.1)')
	parser.add_argument('-t', '--testFrac',  default = 0.1,     help = 'Fraction of events for testing (0.1)')
	parser.add_argument('-b', '--batchSize', default = 500,     help = 'Training batch size (500)')
	parser.add_argument('-l', '--lRate',     default = 0.001,   help = 'Training learning rate (0.001)')
	parser.add_argument('-e', '--epochs',    default = 10,      help = 'Training epochs (10)')
	parser.add_argument('--noHit',  action = 'store_true', 		help = 'Do not use hit channel')
	parser.add_argument('--noTime', action = 'store_true', 		help = 'Do not use time channel')
	parser.add_argument('--norm',   action = 'store_true',		help = 'Normalise the channels')
	parser.add_argument('-s', '--imageSize', default = 32, 		help = 'Input image size (32)')

	# Check arguments
	args = parser.parse_args()

	if args.noHit and args.noTime:
		print("Error: Need to use at least one channel!")
		sys.exit()	
	elif args.train and args.eval or args.train and args.optimise or args.eval and args.optimise:
		print("Error: Can only do one thing at a time!")
		sys.exit()
	elif args.train and args.eval and args.optimise:
		print("Error: Can only do one thing at a time!")
		sys.exit()	
	elif not args.train and not args.eval and not args.optimise:
		print("Error: Need to specify something to do!")
		sys.exit()			

	return args  

###################################################
#             File and Array Handling             #
###################################################

# Load from a single .txt data file generated from cvn_make_images.C into a numpy array
# Splits into the given fractions for [train, validation, testing]
def load_txt_file(file, skipRows, validFrac, testFrac):
	print("load_txt_file ...")

	# Load the txt file into a numpy array
	combined_data = np.loadtxt(file, dtype='float32', skiprows=skipRows)

	np.random.shuffle(combined_data)                        # Shuffle the combined array

	total_events = combined_data.shape[0]                   # Total number of events
	num_training = int(total_events*(1-validFrac-testFrac)) # Number to use for training
	num_validate = int(total_events*validFrac)              # Number to use for validation     

	# Split into training, validation and test samples
	split_data = np.split(combined_data, [num_training, num_training+num_validate, total_events])

	# Print image numbers
	print("Events-> Total:{0} Train:{1} Validate:{2} Test:{3}".format(
		total_events, num_training, num_validate, (total_events-num_training-num_validate)))

	# Return the samples separately (Training, Validation, Testing)
	return split_data[0], split_data[1], split_data[2]    

# Load from a list of .txt data files generated from cvn_make_images.C into a numpy array
# Splits into the given fractions for [train, validation, testing]
def load_multiple_txt_files(fileNames, skipRows, validFrac, testFrac):
	print("load_multiple_txt_files ...")

	# Load the txt files into numpy arrays
	file_arrays = []
	for eventType in range(len(fileNames)):
		file_arrays.append(np.loadtxt(fileNames[eventType], dtype='float32', skiprows=skipRows))

	combined_data = np.concatenate(file_arrays)             # Concatenate them together

	np.random.shuffle(combined_data)                        # Shuffle the combined array

	total_events = combined_data.shape[0]                   # Total number of events
	num_training = int(total_events*(1-validFrac-testFrac)) # Number to use for training
	num_validate = int(total_events*validFrac)              # Number to use for validation     

	# Split into training, validation and test samples
	split_data = np.split(combined_data, [num_training, num_training+num_validate, total_events])

	# Print image numbers
	print("Events-> Total:{0} Train:{1} Validate:{2} Test:{3}".format(
		total_events, num_training, num_validate, (total_events-num_training-num_validate)))

	# Return the samples separately (Training, Validation, Testing)
	return split_data[0], split_data[1], split_data[2]    

# Extract the .txt file data formatting splitting into different numpy arrays and normalise the images
# [Labels(1), Beam Energies(1), Lepton Parameters(7), charge_pixels(32*32), time_pixels(32*32)]
def labels_energies_parameters_images(data, norm, noHit, noTime, imageSize):

	# Size of the image
	imageDataLength = imageSize * imageSize

	# Split the data into the different numpy arrays we need
	labels      = data[:, 0]        			# Index 0
	energies    = data[:, 1]        			# Index 1
	parameters  = data[:, 2:9]      			# Index 2-8
	images_hit  = data[:, 9:imageDataLength+9]  # Index 9-1032
	images_time = data[:, imageDataLength+9:]   # Index 1033-2057

	if norm:
		images_hit  = tf.keras.utils.normalize(images_hit, axis=1)          # Norm hit images
		images_time = tf.keras.utils.normalize(images_time, axis=1)         # Norm time images

	image_shape = (imageSize, imageSize, 1)									# Define single channel shape
	images_hit  = images_hit.reshape(images_hit.shape[0], *image_shape)     # Reshape hit images
	images_time = images_time.reshape(images_time.shape[0], *image_shape)   # Reshape time images

	images = np.concatenate((images_hit,images_time), 3)    				# Combine the channels together

	print("Labels:{0} Energies:{1} Params:{2} Images:{3}".format( 			# Print array shapes
		  labels.shape, energies.shape, parameters.shape, images.shape))    

	# Return depending on which channels are active
	if noHit and not noTime:
		print("Just using time channel!")
		return labels, energies, parameters, images_time
	elif not noHit and noTime:
		print("Just using hit channel!")
		return labels, energies, parameters, images_hit
	else:
		return labels, energies, parameters, images

# Just return the label and images (for cvn_pid_network training)
def labels_images(data, norm, noHit, noTime, imageSize):
	labels, energies, parameters, images = labels_energies_parameters_images(data, norm, noHit, noTime, imageSize)
	return labels, images

# Just return a specific parameter and the images (for cvn_parameter_network training)
def parameter_images(data, norm, noHit, noTime, index, imageSize):
	labels, energies, parameters, images = labels_energies_parameters_images(data, norm, noHit, noTime, imageSize)
	parameter = parameters[:, index]
	return parameter, images	

# Replace all labels in the array with a different one
def replace_labels(array, input, output):
	np.place(array, array==(input), output) # Replace "input" with "output" labels

# Use if you mislabel the categories, starting at 1 not 0
def shift_labels_down(array):
	replace_labels(array, 1.0, 0.0)
	replace_labels(array, 2.0, 1.0)
	replace_labels(array, 3.0, 2.0)
	replace_labels(array, 4.0, 3.0)
	replace_labels(array, 5.0, 4.0)

###################################################
#               Callback Functions                #
###################################################

# Saves the current model state at each epoch
def callback_checkpoint(path):
	return callbacks.ModelCheckpoint(path, save_weights_only=True, verbose=1)

# Stores tensorboard information at each epoch
def callback_tensorboard(logDir):
	return callbacks.TensorBoard(log_dir=logDir, write_graph=True, write_grads=True,
								   histogram_freq=1, write_images = True)

###################################################
#               Plotting Functions                #
###################################################

# pyplot a specific channel array as an image
def plot_image(image, index, size):
	plot = image[index, :].reshape((size,size))
	plt.imshow(plot)
	plt.show()

# pyplot the history of a category based training
def plot_category_history(history):
	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

# pyplot the history of a regression based training
def plot_regression_history(history):
	# summarize history for accuracy
	plt.plot(history.history['mean_absolute_error'])
	plt.plot(history.history['val_mean_absolute_error'])
	plt.title('model error')
	plt.ylabel('mean absolute error')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['mean_squared_error'])
	plt.plot(history.history['val_mean_squared_error'])
	plt.title('model square error')
	plt.ylabel('mean square error')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

# pyplot the (true-estimated) distribution
def plot_true_minus_estimation(diff):
	plt.hist(diff, bins=50)
	#plt.hist(diff, bins=50, range=[-1500, 1500]) # Can specify a range
	plt.ylabel('Events')
	plt.xlabel('True - Estimation')
	plt.show()

###################################################
#                Output Functions                 #
###################################################

# Saves the history of a category based training
def save_category_history(train_history, outputFile):
	print("Output: save_category_history")

	name, ext = os.path.splitext(outputFile)
	outputName = name + "_history.txt"
	output_file = open(outputName, "w")

	acc = train_history.history['acc']
	val_acc = train_history.history['val_acc']
	loss = train_history.history['loss']
	val_loss = train_history.history['val_loss']

	for epoch in range(len(acc)):
		out = str(acc[epoch])+" "+str(val_acc[epoch])+" "+str(loss[epoch])+" "+str(val_loss[epoch])
		out += "\n"
		output_file.write(out)
	output_file.close()  

# Save the test output from a category based model
def save_category_output(categories, test_labels, test_energies, test_parameters, test_output, outputFile):
	print("Output: save_category_output")

	if len(test_labels) != len(test_energies) or len(test_labels) != len(test_output):
		print("Error: Arrays are not the same size")
		sys.exit()	

	# [label, beamE, parameters, testOutput[Outputs for the different categories]]
	output_file = open(outputFile, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]))
		for category in range(categories):
			out += (" " + str(test_output[num][category]))
		out += "\n"
		output_file.write(out)
	output_file.close()     

# Saves the history of a regression based training
def save_regression_history(train_history, outputFile):
	print("Output: save_regression_history")

	name, ext = os.path.splitext(outputFile)
	outputName = name + "_history.txt"
	output_file = open(outputName, "w")

	loss = train_history.history['loss']
	val_loss = train_history.history['val_loss']
	mean_abs_err = train_history.history['mean_absolute_error']
	val_mean_abs_err = train_history.history['val_mean_absolute_error']
	mean_squared_err = train_history.history['mean_squared_error']
	val_mean_squared_err = train_history.history['val_mean_squared_error']

	for epoch in range(len(mean_abs_err)):
		out = str(loss[epoch])+" "+str(val_loss[epoch])+" "+str(mean_abs_err[epoch])+" "+str(val_mean_abs_err[epoch])+" "+str(mean_squared_err[epoch])+" "+str(val_mean_squared_err[epoch])
		out += "\n"
		output_file.write(out)
	output_file.close()  

# Save the test output from a regression based model
def save_regression_output(test_labels, test_energies, test_parameters, test_output, outputFile):
	print("Output: save_regression_output")

	if len(test_labels) != len(test_output):
		print("Error: test_labels and test_output not the same length")
		sys.exit()	

	# [label, beamE, parameters, testOutput]
	output_file = open(outputFile, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]))
		out += (" " + str(test_output[num, 0]))
		out += "\n"
		output_file.write(out)
	output_file.close()     

