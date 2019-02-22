import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt
import sys
import os
from sklearn.preprocessing import StandardScaler

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
		self.data = None				# Data dictionary
		self.scaler = None				# StandardScaler for both channels (hit,time)

	# Loads the data from .txt file splitting up the arrays and places them in a large dictionary
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

		train_labels, train_pars, train_hit_images, train_time_images = self.decode(split_data[0])
		val_labels, val_pars, val_hit_images, val_time_images = self.decode(split_data[1])
		test_labels, test_pars, test_hit_images, test_time_images = self.decode(split_data[2])

		# Set the data dictionary
		self.data = {"train_labels":train_labels, "train_pars":train_pars,
					 "train_hit_images":train_hit_images, "train_time_images":train_time_images,
					 "val_labels":val_labels, "val_pars":val_pars,
					 "val_hit_images":val_hit_images, "val_time_images":val_time_images,
					 "test_labels":test_labels, "test_pars":test_pars,
					 "test_hit_images":test_hit_images, "test_time_images":test_time_images}

		# Fit the Standard Scalers for the two channels just on the training data
		# We do not want the val/test sets to influence the model
		hit_scaler = StandardScaler().fit(train_hit_images)
		time_scaler = StandardScaler().fit(train_time_images)
		self.scaler = [hit_scaler, time_scaler]

	# Decode the .txt format
	# [Labels(1), Parameters(8), charge_pixels(32*32), time_pixels(32*32)]
	# Parameters = [beamE, vtxX, vtxY, vtxZ, vtxT, theta, phi, lepE]
	def decode(self, data):
		image_data_length = self.image_size * self.image_size 	# Size of the image

		labels      = data[:, 0]        						# Index 0
		parameters  = data[:, 1:9]      						# Index 1-8
		hit_images  = data[:, 9:image_data_length+9]  			# Index 9-1032
		time_images = data[:, image_data_length+9:]   			# Index 1033-2057

		return labels, parameters, hit_images, time_images		# Return decoded arrays

	def print(self):
		# Training Data
		print("Training Data -> Labels:{0} Parameters:{1} HitImages:{2} TimeImages:{3}".format(
			self.data["train_labels"].shape, self.data["train_pars"].shape,	
			self.data["train_hit_images"].shape, self.data["train_time_images"].shape))
		# Validation Data
		print("Validation Data -> Labels:{0} Parameters:{1} HitImages:{2} TimeImages:{3}".format(
			self.data["val_labels"].shape, self.data["val_pars"].shape,	
			self.data["val_hit_images"].shape, self.data["val_time_images"].shape))
		# Testing Data
		print("Testing Data -> Labels:{0} Parameters:{1} HitImages:{2} TimeImages:{3}".format(
			self.data["test_labels"].shape, self.data["test_pars"].shape,	
			self.data["test_hit_images"].shape, self.data["test_time_images"].shape))

	# Returns the shape of the images depending on which channels are active
	def get_image_shape(self):
		if self.no_hit or self.no_time:
			return (self.image_size, self.image_size, 1)
		else:
			return (self.image_size, self.image_size, 2)

	# Returns the images (0=train,1=validation,2=test)
	def get_images(self, norm):
		dict_keys = [("train_hit_images", "train_time_images"), 
				("val_hit_images", "val_time_images"), 
				("test_hit_images", "test_time_images")]

		images = []	# Store the different sample images in a list
		image_shape = (self.image_size, self.image_size, 1)	# Single channel shape
		for sample in range(3):
			hit_img = self.data[dict_keys[sample][0]]
			time_img = self.data[dict_keys[sample][1]]
			# Normalise if we need to
			if norm == "ebe":
				hit_img = tf.keras.utils.normalize(hit_img, axis=1)
				time_img = tf.keras.utils.normalize(time_img, axis=1)
			elif norm == "sss":
				hit_img = self.scaler[0].transform(hit_img)
				time_img = self.scaler[1].transform(time_img)
		
			# Reshape and combine the channels
			hit_img = hit_img.reshape(hit_img.shape[0], *image_shape) 
			time_img = time_img.reshape(time_img.shape[0], *image_shape) 
			images.append(np.concatenate((hit_img, time_img), 3))

		return images[0], images[1], images[2] # Train_images, Validation_images, Test_images

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
def save_category_output(categories, test_labels, test_parameters, test_output, output_file):
	print("utils: save_category_output()")

	if len(test_labels) != len(test_parameters) or len(test_labels) != len(test_output):
		print("Error: Arrays are not the same size")
		sys.exit()	

	# [label, beamE, parameters, testOutput[Outputs for the different categories]]
	output = open(output_file, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]) + " " + str(test_parameters[num][7]))
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
def save_regression_output(test_labels, test_parameters, test_output, output_file):
	print("Output: save_regression_output")

	if len(test_labels) != len(test_output):
		print("Error: test_labels and test_output not the same length")
		sys.exit()	

	# [label, beamE, parameters, test_output]
	output = open(output_file, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num])
		out += (" " + str(test_parameters[num][0]) + " " + str(test_parameters[num][1]) + " " + str(test_parameters[num][2]) + " " + str(test_parameters[num][3]))
		out += (" " + str(test_parameters[num][4]) + " " + str(test_parameters[num][5]) + " " + str(test_parameters[num][6]) + " " + str(test_parameters[num][7]))
		out += (" " + str(test_output[num, 0])) + "\n"
		output.write(out)
	output.close()     

