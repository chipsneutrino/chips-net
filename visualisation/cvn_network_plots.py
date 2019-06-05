# Imports
import os.path
import os.path as path
import sys
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras import optimizers

from tensorflow.python.keras import backend as K

import cvn_models as models
import cvn_utilities as utils

import matplotlib.pyplot as plt

import keras

def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + K.epsilon())
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	return x

def normalize(x):
	# utility function to normalize a tensor by its L2 norm
	return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def main():

	# Settings
	imageXSize = 32
	imageYSize = 32
	image_size = (imageXSize, imageYSize, 1)

	layer_name = 'conv2d_4'
	filter_index = 5

	print("Load Model ...")
	model = models.cvn_parameter_model(6, image_size, 0.001)
	model.load_weights("output/parameter_model.ckpt")
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	input_img = model.input # this is the placeholder for the input images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output
	
	print(model.output.shape)

	loss = K.mean(layer_output[:, :, :, filter_index])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads = normalize(grads)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	# step size for gradient ascent
	step = 1.

	# we start from a gray image with some noise
	input_img_data = np.random.random((1, 32, 32, 1)) * 20
	# run gradient ascent for 20 steps
	for i in range(50):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

		print('Current loss value:', loss_value)
		if loss_value <= 0.:
			# some filters get stuck to 0, we can skip them
			break

	img = deprocess_image(input_img_data[0])
	print(img.shape)
	plot = img.reshape((imageXSize,imageYSize))
	print(plot.shape)
	plt.imshow(plot)
	plt.show()
		
if __name__=='__main__':
	main()