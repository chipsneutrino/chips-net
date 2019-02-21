import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler

###################################################
#            Network Argument Handling            #
###################################################

# Parse the command line argument options
def parse_args():
	parser = argparse.ArgumentParser(description='Train/Evaluate/Optimise CHIPS CVN Networks...')

	# Input and output files (history file will have similar name to output file)
	parser.add_argument('input', help = 'Path to combined input "image" .txt file')
	parser.add_argument('-o', '--output',    default='../../output/output.txt', help = 'Output .txt file')
	parser.add_argument('-n', '--network',	 default = 'ppe', 	help = 'Type of network pid or ppe')
	parser.add_argument('-w', '--weights', 	 default='../../output/cp.ckpt',    help = 'Network weights')

	# What function are we doing?
	parser.add_argument('--train', 		action = 'store_true',  help = 'Use to train the network')
	parser.add_argument('--eval',  		action = 'store_true',  help = 'Use to evaluate on the network')
	parser.add_argument('--opt', 	action = 'store_true', 		help = 'Use to optimise the network')

	# Network Hyperparameters
	parser.add_argument('-p', '--parameter',	default = -1,	help = 'Parameter to fit (lepton Energy = 6)')
	parser.add_argument('-v', '--val_frac',   	default = 0.1,  help = 'Fraction of events for validation (0.1)')
	parser.add_argument('-t', '--test_frac',  	default = 0.1,  help = 'Fraction of events for testing (0.1)')
	parser.add_argument('-b', '--batch_size', 	default = 500,  help = 'Training batch size (500)')
	parser.add_argument('-l', '--l_rate',     	default = 0.001,help = 'Training learning rate (0.001)')
	parser.add_argument('-e', '--epochs',    	default = 10,   help = 'Training epochs (10)')
	parser.add_argument('--no_hit',  	action = 'store_true', 	help = 'Do not use hit channel')
	parser.add_argument('--no_time', 	action = 'store_true', 	help = 'Do not use time channel')
	parser.add_argument('--norm',   	action = 'store_true',	help = 'Normalise the channels')
	parser.add_argument('-s', '--image_size', 	default = 32, 	help = 'Input image size (32)')

	# Check arguments
	args = parser.parse_args()

	if args.network != "ppe" and args.network != "pid":
		print("Error: Specify Network!")
		sys.exit()			
	elif args.no_hit and args.no_time:
		print("Error: Need to use at least one channel!")
		sys.exit()	
	elif args.train and args.eval or args.train and args.opt or args.eval and args.opt:
		print("Error: Can only do one thing at a time!")
		sys.exit()
	elif args.train and args.eval and args.opt:
		print("Error: Can only do one thing at a time!")
		sys.exit()	
	elif not args.train and not args.eval and not args.opt:
		print("Error: Need to specify something to do!")
		sys.exit()			

	return args  

###################################################
#                   DataHandler                   #
###################################################

class DataHandler:
	def __init__(self, input_file, val_frac, test_frac, 
				 image_size, no_hit, no_time):
		self.input_file = input_file	# Input .txt file
		self.val_frac = val_frac		# Fraction of events for validation
		self.test_frac = test_frac		# Fraction of events for testing
		self.image_size = image_size	# Image size (32, 64)
		self.no_hit = no_hit			# Don't use hit channel (bool)
		self.no_time = no_time			# Don't use time channel (bool)

		# Data is filled into these three sets in order
		self.train_data = None			# Training data dictionary
		self.val_data = None			# Validation data dictionary
		self.test_data = None			# Testing data dictionary

		self.hit_scaler = None			# Hit channel StandardScaler()
		self.time_scaler = None			# Time channel StandardScaler()

	def load_data(self):
		print("DataHandler: load_data() ...")

		# Load the txt file into a numpy array
		data = np.loadtxt(self.input_file, dtype='float32')

		np.random.shuffle(data) 											# Shuffle the combined array

		total_events = data.shape[0]                   						# Total number of events
		num_training = int(total_events*(1-self.val_frac-self.test_frac)) 	# Number to use for training
		num_validate = int(total_events*self.val_frac)              		# Number to use for validation     

		# Split into training, validation and test samples
		split_data = np.split(data, [num_training, num_training+num_validate, total_events])		

		# Return the samples separately (Training, Validation, Testing)
		self.data_to_dict(split_data[0])
		self.data_to_dict(split_data[1])
		self.data_to_dict(split_data[2])	

	# Extract the .txt file data formatting splitting into different numpy arrays and normalise the images
	# [Labels(1), Beam Energies(1), Lepton Parameters(7), charge_pixels(32*32), time_pixels(32*32)]
	def data_to_dict(self, data):
		# Size of the image
		image_data_length = self.image_size * self.image_size

		# Split the data into the different numpy arrays we need
		labels      = data[:, 0]        				# Index 0
		energies    = data[:, 1]        				# Index 1
		parameters  = data[:, 2:9]      				# Index 2-8
		hit_images  = data[:, 9:image_data_length+9]  	# Index 9-1032
		time_images = data[:, image_data_length+9:]   	# Index 1033-2057

		# Normalise the data
		if self.train_data == None:	# We are looking at the largest set the training data
			self.hit_scaler = StandardScaler().fit(hit_images)
			self.time_scaler = StandardScaler().fit(time_images)		

		hit_images_norm = self.hit_scaler.transform(hit_images)
		time_images_norm = self.time_scaler.transform(time_images)
			
		#hit_images_norm  = tf.keras.utils.normalize(hit_images, axis=1)	# Norm hit images
		#time_images_norm = tf.keras.utils.normalize(time_images, axis=1) # Norm time images

		image_shape = (self.image_size, self.image_size, 1)						# Define single channel shape
		hit_images  = hit_images.reshape(hit_images.shape[0], *image_shape)     # Reshape hit images
		time_images = time_images.reshape(time_images.shape[0], *image_shape)   # Reshape time images
		hit_images_norm  = hit_images_norm.reshape(hit_images_norm.shape[0], *image_shape)     # Reshape hit norm images
		time_images_norm = time_images_norm.reshape(time_images_norm.shape[0], *image_shape)   # Reshape time norm images

		# Set images depending on which channels are active
		images = np.concatenate((hit_images, time_images), 3)    				# Combine the channels together 
		images_norm = np.concatenate((hit_images_norm, time_images_norm), 3)    # Combine the channels together
		if self.no_hit and not self.no_time:
			images = time_images
			images_norm = time_images_norm
		elif not self.no_hit and self.no_time:
			images = hit_images
			images_norm = hit_images_norm

		data_dict = {"labels": labels, "energies": energies, "parameters": parameters, 
					 "images": images, "images_norm": images_norm}

		if self.train_data == None:
			self.train_data = data_dict
		elif self.val_data == None:
			self.val_data = data_dict
		elif self.test_data == None:
			self.test_data = data_dict

		return

	def print(self):
		# Training Data
		print("Training Data -> Labels:{0} Energies:{1} Parameters:{2} Images:{3}".format(
			self.train_data["labels"].shape, self.train_data["energies"].shape,	
			self.train_data["parameters"].shape, self.train_data["images"].shape,	
		))
		# Validation Data
		print("Validation Data -> Labels:{0} Energies:{1} Parameters:{2} Images:{3}".format(
			self.val_data["labels"].shape, self.val_data["energies"].shape,	
			self.val_data["parameters"].shape, self.val_data["images"].shape,	
		))
		# Testing Data
		print("Testing Data -> Labels:{0} Energies:{1} Parameters:{2} Images:{3}".format(
			self.test_data["labels"].shape, self.test_data["energies"].shape,	
			self.test_data["parameters"].shape, self.test_data["images"].shape,	
		))
		# Standard Scalers
		#print(self.hit_scaler.mean_)
		#print(self.time_scaler.mean_)
		#print(self.hit_scaler.var_)
		#print(self.time_scaler.var_)

	def get_image_shape(self):
		if self.no_hit or self.no_time:
			return (self.image_size, self.image_size, 1)
		else:
			return (self.image_size, self.image_size, 2)

###################################################
#               Callback Functions                #
###################################################

# Saves the current model state at each epoch
def callback_checkpoint(path):
	return callbacks.ModelCheckpoint(path, save_weights_only=True, verbose=0)

# Stores tensorboard information at each epoch
def callback_tensorboard(logDir):
	return callbacks.TensorBoard(log_dir=logDir, write_graph=True, write_grads=True,
								   histogram_freq=1, write_images = True)

def callback_early_stop(monitor, delta, epochs):
	return callbacks.EarlyStopping(monitor=monitor, min_delta=delta, patience=epochs,
							  	   verbose=1, mode='min')

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
def save_category_history(train_history, output_file):
	print("utils: save_category_history()")

	name, ext = os.path.splitext(output_file)
	output = open(name + "_history.txt", "w")

	acc = train_history.history['acc']
	val_acc = train_history.history['val_acc']
	loss = train_history.history['loss']
	val_loss = train_history.history['val_loss']

	# [acc, val_acc, loss, val_loss]
	for epoch in range(len(acc)):
		out = str(acc[epoch])+" "+str(val_acc[epoch])+" "+str(loss[epoch])+" "+str(val_loss[epoch])+"\n"
		output.write(out)
	output.close()  

# Save the test output from a category based model
def save_category_output(categories, test_labels, test_energies, test_parameters, test_output, output_file):
	print("utils: save_category_output()")

	if len(test_labels) != len(test_energies) or len(test_labels) != len(test_output):
		print("Error: Arrays are not the same size")
		sys.exit()	

	# [label, beamE, parameters, testOutput[Outputs for the different categories]]
	output = open(output_file, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]))
		for category in range(categories):
			out += (" " + str(test_output[num][category]))
		out += "\n"
		output.write(out)
	output.close()     

# Saves the history of a regression based training
def save_regression_history(train_history, output_file):
	print("Output: save_regression_history")

	name, ext = os.path.splitext(output_file)
	output = open(name + "_history.txt", "w")

	loss = train_history.history['loss']
	val_loss = train_history.history['val_loss']
	mean_abs_err = train_history.history['mean_absolute_error']
	val_mean_abs_err = train_history.history['val_mean_absolute_error']
	mean_squared_err = train_history.history['mean_squared_error']
	val_mean_squared_err = train_history.history['val_mean_squared_error']

	# [loss, val_loss, mean_abs_err, val_mean_abs_err, mean_squared_err, val_mean_squared_err]
	for epoch in range(len(mean_abs_err)):
		out = str(loss[epoch])+" "+str(val_loss[epoch])+" "+str(mean_abs_err[epoch])+" "+str(val_mean_abs_err[epoch])+" "+str(mean_squared_err[epoch])+" "+str(val_mean_squared_err[epoch])+"\n"
		output.write(out)
	output.close()  

# Save the test output from a regression based model
def save_regression_output(test_labels, test_energies, test_parameters, test_output, output_file):
	print("Output: save_regression_output")

	if len(test_labels) != len(test_output):
		print("Error: test_labels and test_output not the same length")
		sys.exit()	

	# [label, beamE, parameters, test_output]
	output = open(output_file, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]))
		out += (" " + str(test_output[num, 0])) + "\n"
		output.write(out)
	output.close()     

