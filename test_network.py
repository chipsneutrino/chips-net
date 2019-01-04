# Imports
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import optimizers

import models
import utils

# Input txt files
'''
imageFiles = ["../data/nuel_cc_qe_test64.txt",      #6000 events
              "../data/nuel_cc_nonqe_test64.txt",   #6000 events
              "../data/numu_cc_qe_test64.txt",      #6000 events
              "../data/numu_cc_nonqe_test64.txt",   #6000 events
              "../data/all_nc_test64.txt"]          #6000 events
imageFiles = ["../data/nuel_cc_qe_test.txt",        #6000 events
              "../data/nuel_cc_nonqe_test.txt",     #6000 events
              "../data/numu_cc_qe_test.txt",        #6000 events
              "../data/numu_cc_nonqe_test.txt",     #6000 events
              "../data/all_nc_test.txt"]            #6000 events
'''
imageFiles = ["../data/nuel_cc_qe_test_v2.txt",        #6000 events
              "../data/nuel_cc_nonqe_test_v2.txt",     #6000 events
              "../data/numu_cc_qe_test_v2.txt",        #6000 events
              "../data/numu_cc_nonqe_test_v2.txt",     #6000 events
              "../data/all_nc_test_v2.txt"]            #6000 events
'''
imageFiles = ["../data/nuel_cc_qe_images.txt",      #150000 events
              "../data/nuel_cc_nonqe_images.txt",   #150000 events
              "../data/numu_cc_qe_images.txt",      #150000 events
              "../data/numu_cc_nonqe_images.txt",   #150000 events
              "../data/all_nc_images.txt"]          #150000 events
'''

###
# - Get the Nova googleNet working
# - Energy estimation as well!!!
###

### STANFORD
# - Subtract the mean image across the whole sample set (train, val, test) from each image. When I find it, hopefully it is already centered!!!
# - Subtract the per-channel mean from every channel (hit,time) so it has "zero mean"
# - Look at how we are initializing the weights, it seems it may be important  

# Image settings
categories      = 5
skip_rows       = 0
validate_frac   = 0.1
testing_frac    = 0.1
norm_images     = True
image_shape = (32, 32, 1)
input_shape = (32, 32, 2)

# Training settings
batch_size      = 500
learning_rate   = 0.001
epochs          = 20
checkpoint_path = "model.ckpt"
tensorboard_dir = "tb_dir"

# Output Settings
outputFile = "5_2.txt"

def main():
    print("#### chips_cvn - test_network ####")

    # Load the data and split into the different samples we need
    train_data, validate_data, test_data = utils.load_txt_files_and_split(
        imageFiles, skip_rows, validate_frac, testing_frac)

    # Split the training and validation data into labels and images and normalize
    train_labels, train_images = utils.labels_images(train_data, norm_images, image_shape)

    validate_labels, validate_images = utils.labels_images(validate_data, norm_images, image_shape)

    # Get the keras model and print its summary
    cnn_model = models.cnn_model(categories, input_shape)
    cnn_model.summary()

    # Compile the model
    cnn_model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.Adam(lr=learning_rate),
                      metrics=['accuracy'])    

    # Fit the model
    cnn_model.fit(train_images, train_labels, batch_size=batch_size,
                  epochs=epochs, verbose=1,
                  validation_data=(validate_images, validate_labels),
                  callbacks=[utils.callback_checkpoint(checkpoint_path)])

    #cnn_model.load_weights(checkpoint_path)

    # Split the testing data into labels, beam energies and images
    test_labels, test_energies, test_images = utils.labels_energy_images(test_data, norm_images, image_shape)

    # Score the model on the test sample
    score = cnn_model.evaluate(test_images, test_labels, verbose=0)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(score[0], score[1]))

    # Evaluate the test sample
    output = cnn_model.predict(test_images, verbose=0)

    # Save the test sample predictions to file for further analysis
    output_file = open(outputFile, "w")
    for num in range(len(test_labels)):
        out = str(test_labels[num]) + " " + str(test_energies[num])
        for category in range(categories):
            out += (" " + str(output[num][category]))
        out += "\n"
        output_file.write(out)
    output_file.close() 

if __name__=='__main__':
	main()