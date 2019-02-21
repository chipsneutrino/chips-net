# The implementation of the PID and Parameter Estimation Convolutional Visual Networks
import os
import talos as ta

import models
import utils

class Network(): # Parent Network Class
	def __init__(self, data_handler, output_name, batch_size,
				 learning_rate, num_epochs, norm, weights):
		self.data = data_handler
		self.output_name = output_name
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.norm = norm
		self.weights = weights

class PIDNetwork(Network): # Particle Identification (PID) Network Class
	def __init__(self, data_handler, output_name, batch_size, 
				 learning_rate, num_epochs, norm, weights):
		Network.__init__(self, data_handler, output_name, batch_size, 
						 learning_rate, num_epochs, norm, weights)

	def train(self):
		print("PIDNetwork: train() ...")
		print("Output:{0} Weights:{1}".format(self.output_name, self.weights)) 
		print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
			  self.norm, self.batch_size, self.learning_rate, self.num_epochs))

		# Get the specific Keras parameter model
		categories = 5
		model = models.pid_model(categories, self.data.get_image_shape(), self.learning_rate)	

		# Get the images from the different samples
		train_data = self.data.train_data["images_norm"]
		val_data = self.data.val_data["images_norm"]
		test_data = self.data.test_data["images_norm"]
		if not self.norm:
			train_data = self.data.train_data["images"]
			val_data = self.data.val_data["images"]
			test_data = self.data.test_data["images"]			

		# Fit the model with the training data and store the training "history"
		train_history = model.fit(train_data, self.data.train_data["labels"],
								  batch_size=self.batch_size, epochs=self.num_epochs, verbose=1,
								  validation_data=(val_data, self.data.val_data["labels"]),
								  callbacks=[utils.callback_checkpoint(self.weights),
								  			 utils.callback_early_stop("val_acc", 0.01, 5)])    

		# Score the model
		score = model.evaluate(test_data, self.data.test_data["labels"], verbose=0)
		print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(score[0], score[1]))	

		# Save history to training output
		utils.save_category_history(train_history, self.output_name)	

		# Evaluate the test sample on the trained model and save output to file
		test_output = model.predict(test_data, verbose=0)
		utils.save_category_output(categories, test_labels, test_energies, 
								   test_parameters, test_output, self.output_name)

	def evaluate(self):
		print("PIDNetwork: evaluate() ...")

	def optimise(self):
		print("PIDNetwork: optimise() ...")

		x = self.data.train_data["images_norm"]
		y = self.data.train_data["labels"]

		if not self.norm:
			x = self.data.train_data["images"]

		p = {"learning_rate":[0.001], "batch_size":[500], "epochs":[40],
			 "filters_1":[64], "size_1":[3], "pool_1":[2],
			 "filters_2":[128], "size_2":[3], "pool_2":[2],
			 "filters_3":[256], "size_3":[3], "pool_3":[2],
			 "dense":[512], "dropout":[0.2], "stop_size":[0.01], "stop_epochs":[5],
			 "categories":[5]}

		t = ta.Scan(x=x, y=y, model=models.pid_model_fit, grid_downsample=1.0, 
					params=p, dataset_name='pid_model', experiment_no='0',
					clear_tf_session=False, val_split=0.2)

class PPENetwork(Network): # Particle Parameter Estimation (PPE) Network Class
	def __init__(self, data_handler, output_name, batch_size, 
				 learning_rate, num_epochs, norm, weights, parameter):
		Network.__init__(self, data_handler, output_name, batch_size, 
					     learning_rate, num_epochs, norm, weights)
		self.parameter = parameter

	# Train the PPENetwork is self.parameter = -1 we will train all the parameter networks
	def train(self):
		if self.parameter != -1:
			self.train_parameter()
		else:
			name, ext = os.path.splitext(self.output_name)			
			weight_name, weight_ext = os.path.splitext(self.weights)		
			for par in range(7):
				# Change the output file name everytime
				self.output_name = name + "_" + str(par) + ".txt"	
				self.weights = weight_name + "_" + str(par) + ".ckpt"

				# Don't use the normalised images for vtxT and track energy
				if par == 3 or par == 6:
					self.norm = False	
				else:
					self.norm = True

				# Set the parameter to train
				self.parameter = par
				self.train_parameter()				

	def train_parameter(self):
		print("PPENetwork: train_parameter() ...")
		print("Parameter:{0}".format(self.parameter))
		print("Output:{0} Weights:{1}".format(self.output_name, self.weights)) 
		print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
			  self.norm, self.batch_size, self.learning_rate, self.num_epochs))

		# Get the specific Keras parameter model
		model = models.parameter_model(self.parameter, self.data.get_image_shape(), 
										   self.learning_rate)

		# Get the images from the different samples
		train_data = self.data.train_data["images_norm"]
		val_data = self.data.val_data["images_norm"]
		test_data = self.data.test_data["images_norm"]
		if not self.norm:
			train_data = self.data.train_data["images"]
			val_data = self.data.val_data["images"]
			test_data = self.data.test_data["images"]			

		# Fit the model with the training data and store the training "history"
		train_history = model.fit(train_data, self.data.train_data["parameters"][:, self.parameter],
								  batch_size=self.batch_size, epochs=self.num_epochs, verbose=1,
								  validation_data=(val_data, self.data.val_data["parameters"][:, self.parameter]),
								  callbacks=[utils.callback_checkpoint(self.weights),
								  			 utils.callback_early_stop("val_mean_absolute_error", 5, 5)])
		
		trained_epochs = len(train_history.history['loss'])
		print(trained_epochs)

		if self.output_name is not None:
			# Save history to training output
			utils.save_regression_history(train_history, self.output_name)

			# Evaluate the test sample on the trained model and save output to file
			test_output = model.predict(test_data, verbose=0)
			utils.save_regression_output(self.data.test_data["labels"], self.data.test_data["energies"], 
										 self.data.test_data["parameters"], test_output, self.output_name)

		return train_history.history['val_mean_absolute_error'][trained_epochs-1]        

	def evaluate(self):
		print("PPENetwork: evaluate() ...")

	def optimise(self):
		print("PPENetwork: optimise() ...")

		x = self.data.train_data["images_norm"], self.data.val_data["images_norm"]
		y = self.data.train_data["parameters"][:, self.parameter]
		if not self.norm:
			x = self.data.train_data["images"]

		p = {"learning_rate":[0.001], "batch_size":[500], "epochs":[40],
			 "filters_1":[32], "size_1":[3], "pool_1":[2],
			 "filters_2":[64], "size_2":[3], "pool_2":[2],
			 "dense":[128], "dropout":[0.0], "stop_size":[5], "stop_epochs":[5]}

		t = ta.Scan(x=x, y=y, model=models.parameter_model_fit, grid_downsample=1.0, 
					params=p, dataset_name='parameter_model', experiment_no='0',
					clear_tf_session=False, val_split=0.2)

def main():
	args = utils.parse_args() # Get the command line arguments

	# Load the data into the DataHandler and print the summary
	data = utils.DataHandler(args.input, float(args.val_frac), float(args.test_frac),
					   		 int(args.image_size), args.no_hit, args.no_time)
	data.load_data()
	data.print()

	if args.network == "ppe":
		network = PPENetwork(data, args.output, int(args.batch_size), float(args.l_rate), 
							 int(args.epochs), args.norm, args.weights, int(args.parameter))
		if args.train:
			network.train()
		elif args.eval:
			network.evaluate()
		elif args.opt:
			network.optimise()
		
	elif args.network == "pid":
		network = PIDNetwork(data, args.output, int(args.batch_size), float(args.l_rate), 
							 int(args.epochs), args.norm, args.weights)
		if args.train:
			network.train()
		elif args.eval:
			network.evaluate()
		elif args.opt:
			network.optimise()	

if __name__=='__main__':
	main()