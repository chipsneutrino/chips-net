import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

def cvn_pid_model(categories, inputShape, learningRate):
	pid_model = tf.keras.Sequential([
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

	pid_model.summary()                                                 # Print the model summary
	pid_model.compile(loss='sparse_categorical_crossentropy',           # Compile the model
					  optimizer=optimizers.Adam(lr=learningRate),
					  metrics=['accuracy']) 
	return pid_model

def cvn_parameter_model(parameter, inputShape, learningRate):

	if parameter == 0:
		# Vertex x-position model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 1:
		# Vertex y-position model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 2:
		# Vertex z-position model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 3:
		# Vertex time model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 4:
		# Track theta-dir model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 5:
		# Track phi-dir model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	elif parameter == 6:
		# Track energy model
		parameter_model = tf.keras.Sequential([
			layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=inputShape),
			layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
			layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
			layers.MaxPooling2D(pool_size=2),
			layers.Flatten(),
			layers.Dense(128, activation='relu'),
			layers.Dense(1, activation='linear')
		])

		parameter_model.summary()                                           	# Print the model summary
		parameter_model.compile(loss='mean_squared_error',        				# Compile the model
							optimizer=optimizers.Adam(lr=float(learningRate)),
							metrics=['mae', 'mse']) 
		return parameter_model

	else:
		print("Error: Do not recognise parameter number!")
		sys.exit()	
