from vis.utils import utils
from vis.visualization import visualize_activation
from vis.input_modifiers import Jitter
from vis.visualization import get_num_filters
from keras import activations

import numpy as np

import cvn_models as models

from matplotlib import pyplot as plt

model = models.cvn_pid_model(5, (32, 32, 2), 0.001)
model.load_weights("output/pid_model_q.ckpt")

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, "dense_2")

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)

'''
for cat in range(5):
    img = visualize_activation(model, layer_idx, filter_indices=cat, max_iter=200, input_modifiers=[Jitter(4)])
    plot = img[:,:,0].reshape((32,32))
    plt.imshow(plot)
    plt.show()
    plot = img[:,:,1].reshape((32,32))
    plt.imshow(plot)
    plt.show()
'''

# The name of the layer we want to visualize
# You can see this in the model definition.
layer_name = 'conv2d_6'
layer_idx = utils.find_layer_idx(model, layer_name)

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter.
vis_images = []
for idx in range(16):
    img = visualize_activation(model, layer_idx, filter_indices=idx)
    
    # Utility to overlay text on image.
    #img = utils.draw_text(img, 'Filter {}'.format(idx))    
    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=8)
print(stitched.shape)    
plt.axis('off')
plt.imshow(stitched[:,:,0])
plt.title(layer_name)
plt.show()