# -*- coding: utf-8 -*-

"""
Explanation classes for exploring how the models work

This module contains all the explanation classes used to explain how
the models work. They aim to provide insight into the inner workings
of how the models are achieving their task.
"""

from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np


class GradCAM:
    """Class implementing GradCAM algorithm."""

    def __init__(self, model, classIdx, layerName):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

    def compute_heatmap(self, inputs, eps=1e-8):
        """Class implementing GradCAM algorithm."""
        # construct our gradient model to additionally return last conv layer outputs
        gradModel = Model(inputs=[self.model.inputs],
                          outputs=[self.model.get_layer(self.layerName).output,
                                   self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Pass the inputs through the gradient model, and grab the loss
            # associated with the specific class index
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        #(w, h) = (image.shape[2], image.shape[1])
        #heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        #numer = heatmap - np.min(heatmap)
        #denom = (heatmap.max() - heatmap.min()) + eps
        #heatmap = numer / denom
        #heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        # return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        #heatmap = cv2.applyColorMap(heatmap, colormap)
        #output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        # return (heatmap, output)
        return 1
