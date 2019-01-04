import numpy as np
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt

def load_txt_files_and_split(fileNames, skipRows, validFrac, testFrac):
    print("Loading Image Files...")

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

def labels_images(data, norm, imageShape):
    # Format of input arrays = [labels, beamE, imagePixels...]
    labels      = data[:, 0]
    images_hit  = data[:, 2:1026]
    images_time = data[:, 1026:]

    if norm:
        images_hit  = tf.keras.utils.normalize(images_hit, axis=1)          # Normalize the images
        images_time = tf.keras.utils.normalize(images_time, axis=1)         # Normalize the images

    images_hit  = images_hit.reshape(images_hit.shape[0], *imageShape)      # Reshape the images
    images_time = images_time.reshape(images_time.shape[0], *imageShape)    # Reshape the images

    images = np.concatenate((images_hit,images_time), 3)

    print("Shapes-> Labels:{0} Images:{1}".format( # Print array shapes
        labels.shape, images.shape))

    return labels, images

def labels_energy_images(data, norm, imageShape):
    # Format of input arrays = [labels, beamE, imagePixels...]
    labels      = data[:, 0]
    energies    = data[:, 1]
    images_hit  = data[:, 2:1026]
    images_time = data[:, 1026:]

    if norm:
        images_hit = tf.keras.utils.normalize(images_hit, axis=1)   # Normalize the images
        images_time = tf.keras.utils.normalize(images_time, axis=1)         # Normalize the images

    images_hit = images_hit.reshape(images_hit.shape[0], *imageShape)   # Reshape the images
    images_time = images_time.reshape(images_time.shape[0], *imageShape)    # Reshape the images

    images = np.concatenate((images_hit,images_time), 3)

    print("Shapes-> Labels:{0} Energies:{1} Images:{2}".format( # Print array shapes
        labels.shape, energies.shape, images.shape))

    return labels, energies, images

def replace_labels(array, input, output):
    np.place(array, array==(input), output) # Replaces all labels of type "input" with "output"

def callback_tensorboard(logDir):
    tensorboard = callbacks.TensorBoard(log_dir=logDir, write_graph=True, write_grads=True,
        histogram_freq=1, write_images = True)
    return tensorboard

def callback_checkpoint(path):
    checkpoint = callbacks.ModelCheckpoint(path, save_weights_only=True, verbose=1)
    return checkpoint

def plot_image(imageArray, index):
    image = imageArray[index, :].reshape((32,32))
    plt.imshow(image)
    plt.show()