import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt

###################################################
#             File and Array Handling             #
###################################################

image_shape = (32, 32, 1)

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
def labels_energies_parameters_images(data, norm):
    labels      = data[:, 0]        # Index 0
    energies    = data[:, 1]        # Index 1
    parameters  = data[:, 2:9]      # Index 2-8
    images_hit  = data[:, 9:1033]   # Index 9-1032
    images_time = data[:, 1033:]    # Index 1033-2057

    #TODO: Think about improving upon this simple normalisation
    if norm:
        images_hit  = tf.keras.utils.normalize(images_hit, axis=1)          # Norm hit images
        images_time = tf.keras.utils.normalize(images_time, axis=1)         # Norm time images

    images_hit  = images_hit.reshape(images_hit.shape[0], *image_shape)     # Reshape hit images
    images_time = images_time.reshape(images_time.shape[0], *image_shape)   # Reshape time images

    images = np.concatenate((images_hit,images_time), 3)    # Combine the two channels together

    print("Labels:{0} Energies:{1} Params:{2} Images:{3}".format( # Print array shapes
          labels.shape, energies.shape, parameters.shape, images.shape))    

    return labels, energies, parameters, images

# Just return the labels and images
def labels_images(data, norm):
    labels, energies, parameters, images = labels_energies_parameters_images(data, norm)
    return labels, images

# Just return the labels, beam energies and images
def labels_energies_images(data, norm):
    labels, energies, parameters, images = labels_energies_parameters_images(data, norm)
    return labels, energies, images

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

def callback_log_epoch(logFile):
    return logFile

###################################################
#               Plotting Functions                #
###################################################

def plot_image(image, index):
    plot = image[index, :].reshape((32,32))
    plt.imshow(plot)
    plt.show()

