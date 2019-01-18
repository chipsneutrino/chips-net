import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt
import os

###################################################
#             File and Array Handling             #
###################################################

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

# Loads the "image" .txt files and splits the combined and shuffled array
# into the given fractions for [train, validation, testing]
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

# Format of image .txt files is either...
# [Labels(1), Beam Energies(1), Lepton Parameters(7), pixelsH(1024), pixelsT(1024)]
# Extract this formatting and normalise images if required
def labels_energies_parameters_images(data, norm, noHit, noTime, imageSize):

	imageDataLength = imageSize * imageSize

	labels      = data[:, 0]        # Index 0
	energies    = data[:, 1]        # Index 1
	parameters  = data[:, 2:9]      # Index 2-8
	images_hit  = data[:, 9:imageDataLength+9]   # Index 9-1032
	images_time = data[:, imageDataLength+9:]    # Index 1033-2057

	#TODO: Think about improving upon this simple normalisation
	if norm:
		images_hit  = tf.keras.utils.normalize(images_hit, axis=1)          # Norm hit images
		images_time = tf.keras.utils.normalize(images_time, axis=1)         # Norm time images

	image_shape = (imageSize, imageSize, 1)

	images_hit  = images_hit.reshape(images_hit.shape[0], *image_shape)     # Reshape hit images
	images_time = images_time.reshape(images_time.shape[0], *image_shape)   # Reshape time images

	images = np.concatenate((images_hit,images_time), 3)    # Combine the two channels together

	print("Labels:{0} Energies:{1} Params:{2} Images:{3}".format( # Print array shapes
		  labels.shape, energies.shape, parameters.shape, images.shape))    

	if noHit and not noTime:
		print("Just using time channel!")
		return labels, energies, parameters, images_time
	elif not noHit and noTime:
		print("Just using hit channel!")
		return labels, energies, parameters, images_hit
	else:
		return labels, energies, parameters, images

# Just return the labels and images
def labels_images(data, norm, noHit, noTime, imageSize):
	labels, energies, parameters, images = labels_energies_parameters_images(data, norm, noHit, noTime, imageSize)
	return labels, images

# Just return the labels, beam energies and images
def labels_energies_images(data, norm, noHit, noTime, imageSize):
	labels, energies, parameters, images = labels_energies_parameters_images(data, norm, noHit, noTime, imageSize)
	return labels, energies, images

def parameter_images(data, norm, noHit, noTime, index, imageSize):
	labels, energies, parameters, images = labels_energies_parameters_images(data, norm, noHit, noTime, imageSize)
	parameter = parameters[:, index]
	return parameter, images	

def replace_labels(array, input, output):
	np.place(array, array==(input), output) # Replace "input" with "output" labels

def shift_labels_down(array):
	replace_labels(array, 1.0, 0.0)
	replace_labels(array, 2.0, 1.0)
	replace_labels(array, 3.0, 2.0)
	replace_labels(array, 4.0, 3.0)
	replace_labels(array, 5.0, 4.0)

###################################################
#               Callback Functions                #
###################################################

def callback_checkpoint(path):
	return callbacks.ModelCheckpoint(path, save_weights_only=True, verbose=1)

def callback_tensorboard(logDir):
	return callbacks.TensorBoard(log_dir=logDir, write_graph=True, write_grads=True,
								   histogram_freq=1, write_images = True)

###################################################
#               Plotting Functions                #
###################################################

def plot_image(image, index, size):
	plot = image[index, :].reshape((size,size))
	plt.imshow(plot)
	plt.show()

def plot_category_history(history):
	# list all data in history
	#print(history.history.keys())
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

def plot_regression_history(history):
	# list all data in history
	print(history.history.keys())
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

def plot_true_minus_estimation(diff):
	plt.hist(diff, bins=50)
	#plt.hist(diff, bins=50, range=[-1500, 1500])
	plt.ylabel('Events')
	plt.xlabel('True - Estimation')
	plt.show()

###################################################
#                Output Functions                 #
###################################################

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

	#plot_category_history(train_history) # TEST PLOT 

def save_category_output(categories, test_labels, test_energies, test_output, outputFile):
	print("Output: save_category_output")

	if len(test_labels) != len(test_energies) or len(test_labels) != len(test_output):
		print("Error: Arrays are not the same size")
		sys.exit()	

	output_file = open(outputFile, "w")
	for num in range(len(test_labels)):
		out = str(test_labels[num]) + " " + str(test_energies[num])
		for category in range(categories):
			out += (" " + str(test_output[num][category]))
		out += "\n"
		output_file.write(out)
	output_file.close()     

def save_regression_history(train_history, outputFile):
	print("Output: save_regression_history")

	name, ext = os.path.splitext(outputFile)
	outputName = name + "_history.txt"
	output_file = open(outputName, "w")

	mean_abs_err = train_history.history['mean_absolute_error']
	val_mean_abs_err = train_history.history['val_mean_absolute_error']
	mean_squared_err = train_history.history['mean_squared_error']
	val_mean_squared_err = train_history.history['val_mean_squared_error']

	for epoch in range(len(mean_abs_err)):
		out = str(mean_abs_err[epoch])+" "+str(val_mean_abs_err[epoch])+" "+str(mean_squared_err[epoch])+" "+str(val_mean_squared_err[epoch])
		out += "\n"
		output_file.write(out)
	output_file.close()  

	#plot_regression_history(train_history) # TEST PLOT

def save_regression_output(test_labels, test_output, outputFile):
	print("Output: save_regression_output")

	if len(test_labels) != len(test_output):
		print("Error: test_labels and test_output not the same length")
		sys.exit()	

	# Write [true, estimation, true-estimation] to file for every event
	output_file = open(outputFile, "w")
	true_minus_estimation = []
	for num in range(len(test_labels)):
		diff = test_labels[num] - test_output[num, 0]
		out = str(test_labels[num]) + " " + str(test_output[num, 0]) + " " + str(diff)
		out += "\n"
		output_file.write(out)
		true_minus_estimation.append(diff)
	output_file.close()     
	#plot_true_minus_estimation(true_minus_estimation) # TEST PLOT


