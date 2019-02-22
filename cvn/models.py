import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import optimizers
import utils

def pid_model(categories, input_shape, learning_rate):

	# Structure the sequential model
	model = tf.keras.Sequential([
		layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
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

	# Print the model summary
	model.summary()

	# Compile the model
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer=optimizers.Adam(lr=float(learning_rate)),
				  metrics=['accuracy']) 

	# Return the compiled model
	return model

# Used for talos optimisation of the pid model
def pid_model_fit(x_train, y_train, x_val, y_val, params):

	# Structure the sequential model
	model = tf.keras.Sequential([
		layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"], padding='same', activation='relu', input_shape=(32,32,2)),
		layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"], activation='relu'),
		layers.MaxPooling2D(pool_size=params["pool_1"]),
		layers.Dropout(params["dropout"]),

		layers.Conv2D(filters=params["filters_2"], kernel_size=params["size_2"], padding='same', activation='relu'),
		layers.Conv2D(filters=params["filters_2"], kernel_size=params["size_2"], activation='relu'),
		layers.MaxPooling2D(pool_size=params["pool_2"]),
		layers.Dropout(params["dropout"]),

		layers.Conv2D(filters=params["filters_3"], kernel_size=params["size_3"], padding='same', activation='relu'),
		layers.Conv2D(filters=params["filters_3"], kernel_size=params["size_3"], activation='relu'),
		layers.MaxPooling2D(pool_size=params["pool_3"]),
		layers.Dropout(params["dropout"]),

		layers.Flatten(),
		layers.Dense(params["dense"], activation='relu'),
		layers.Dropout(params["dropout"]),
		layers.Dense(params["categories"], activation='softmax')
	])

	# Compile the model
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer=optimizers.Adam(lr=params["learning_rate"]),
				  metrics=['accuracy']) 

	# Fit the model
	history = model.fit(x_train, y_train, batch_size=params["batch_size"], 
						epochs=params["epochs"], verbose=1, validation_data=(x_val, y_val),
						callbacks=[utils.callback_early_stop("val_acc", params["stop_size"], params["stop_epochs"])])
	
	# Finally we return the history object and the model
	return history, model

def parameter_model(parameter, input_shape, learning_rate):

	# TODO: Implement different tuned models for each parameter

	# Structure the sequential model
	model = tf.keras.Sequential([
		layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
		layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
		layers.MaxPooling2D(pool_size=2),
		layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
		layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
		layers.MaxPooling2D(pool_size=2),
		layers.Flatten(),
		layers.Dense(64, activation='relu'),
		layers.Dropout(0.0),
		layers.Dense(1, activation='linear')
	])

	# Print the model summary
	model.summary()

	# Compile the model
	model.compile(loss='mean_squared_error',
				  optimizer=optimizers.Adam(lr=float(learning_rate)),
				  metrics=['mae', 'mse']) 

	# Return the compiled model
	return model

# Used for talos optimisation of parameter models
def parameter_model_fit(x_train, y_train, x_val, y_val, params):

	# Structure the sequential model
	model = tf.keras.Sequential([
		layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"], padding='same', activation='relu', input_shape=(32,32,2)),
		layers.Conv2D(filters=params["filters_1"], kernel_size=params["size_1"], activation='relu'),
		layers.MaxPooling2D(pool_size=params["pool_1"]),
		layers.Conv2D(filters=params["filters_2"], kernel_size=params["size_2"], padding='same', activation='relu'),
		layers.Conv2D(filters=params["filters_2"], kernel_size=params["size_2"], activation='relu'),
		layers.MaxPooling2D(pool_size=params["pool_2"]),
		layers.Flatten(),
		layers.Dense(params["dense"], activation='relu'),
		layers.Dropout(params["dropout"]),
		layers.Dense(1, activation='linear')
	])

	# Compile the model
	model.compile(loss='mean_squared_error',
				  optimizer=optimizers.Adam(lr=params["learning_rate"]),
				  metrics=['mae', 'mse']) 

	# Fit the model
	history = model.fit(x_train, y_train, batch_size=params["batch_size"], 
						epochs=params["epochs"], verbose=1, validation_data=(x_val, y_val),
						callbacks=[utils.callback_early_stop("val_mean_absolute_error", params["stop_size"], params["stop_epochs"])])
	
	# Finally we return the history object and the model
	return history, model