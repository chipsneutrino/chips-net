import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

def cnn_model(categories, inputShape, learningRate):
	cnn_model = tf.keras.Sequential([
		layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
		layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
		layers.MaxPooling2D(pool_size=2),
		layers.Dropout(0.2),

		layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
		layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
		layers.MaxPooling2D(pool_size=2),
		layers.Dropout(0.2),

		layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
		layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
		layers.MaxPooling2D(pool_size=2),
		layers.Dropout(0.2),

		layers.Flatten(),
		layers.Dense(512, activation='relu'),
		layers.Dropout(0.2),
		layers.Dense(categories, activation='softmax')
	])

	cnn_model.summary()                                                 # Print the model summary
	cnn_model.compile(loss='sparse_categorical_crossentropy',           # Compile the model
					  optimizer=optimizers.Adam(lr=float(learningRate)),
					  metrics=['accuracy']) 
	return cnn_model
