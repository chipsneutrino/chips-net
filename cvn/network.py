# The implementation of the PID and Parameter Estimation Convolutional Visual Networks
import os
import sys
import csv
import talos as ta
import argparse
import numpy as np

import models
import utils

class Network(): # Parent Network Class
	def __init__(self, data_handler, output_name, settings):
		self.data = data_handler
		self.output_name = output_name
		self.settings = settings

class PIDNetwork(Network): # Particle Identification (PID) Network Class
	def __init__(self, data_handler, output_name, settings):
		Network.__init__(self, data_handler, output_name, settings)

	def train(self):
		print("PIDNetwork: train() ...")
		print("Output:{0} Weights:{1}".format(self.output_name, self.settings["weights"])) 
		print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
			  self.settings["norm"], self.settings["batch_size"], self.settings["learning_rate"], self.settings["epochs"]))

		# Get the specific Keras parameter model
		categories = 5
		model = models.pid_model(categories, self.data.get_image_shape(), self.settings["learning_rate"])	

		# Get the images from the different samples
		train_images, val_images, test_images = self.data.get_images(self.settings["norm"])		

		# Fit the model with the training data and store the training "history"
		train_history = model.fit(train_images, 
								  self.data.data["train_labels"],
								  batch_size=int(self.settings["batch_size"]), 
								  epochs=int(self.settings["epochs"]),
								  verbose=1, 
								  validation_data=(val_images, self.data.data["val_labels"]),
								  callbacks=[utils.callback_checkpoint(self.settings["weights"]),
								  			 utils.callback_early_stop("val_acc", 
											   						   self.settings["stop_size"], 
											   						   self.settings["stop_epochs"])])    

		# Score the model
		score = model.evaluate(test_images, self.data.data["test_labels"], verbose=0)
		print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(score[0], score[1]))	

		# Save history
		utils.save_category_history(train_history, self.output_name)	

		# Evaluate the test sample on the trained model and save output to file
		test_output = model.predict(test_images, verbose=0)
		utils.save_category_output(categories, self.data.data["test_labels"], 
								   self.data.data["test_pars"], 
								   test_output, self.output_name)

	def evaluate(self):
		print("PIDNetwork: evaluate() ...")

	def optimise(self):
		print("PIDNetwork: optimise() ...")

		x, val_images, test_images = self.data.get_images(self.norm)		
		y = self.data.data["train_labels"]		

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
	def __init__(self, data_handler, output_name, settings, parameter):
		Network.__init__(self, data_handler, output_name, settings)
		self.parameter = parameter

	# Train the PPENetwork is self.parameter = -1 we will train all the parameter networks
	def train(self):
		if self.parameter != -1:
			self.train_parameter()
		else:
			name, ext = os.path.splitext(self.output_name)				
			for par in range(8):
				# Change the output file name everytime
				self.output_name = name + "_" + str(par) + ".txt"	

				# Set the parameter to train
				self.parameter = par
				self.train_parameter()				

	def train_parameter(self):
		print("PPENetwork: train_parameter() ...")
		print("Parameter:{0}".format(self.parameter))
		print("Output:{0} Weights:{1}".format(self.output_name, self.settings[self.parameter]["weights"])) 
		print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
			  self.settings[self.parameter]["norm"], self.settings[self.parameter]["batch_size"],
			  self.settings[self.parameter]["learning_rate"], self.settings[self.parameter]["epochs"]))

		# Get the specific Keras parameter model
		model = models.ppe_model(self.parameter, self.data.get_image_shape(), 
									   self.settings[self.parameter]["learning_rate"])

		# Get the images from the different samples
		train_images, val_images, test_images = self.data.get_images(self.settings[self.parameter]["norm"])				

		# Fit the model with the training data and store the training "history"
		train_history = model.fit(train_images, 
								  self.data.data["train_pars"][:, self.parameter],
								  batch_size=int(self.settings[self.parameter]["batch_size"]), 
								  epochs=int(self.settings[self.parameter]["epochs"]), 
								  verbose=1,
								  validation_data=(val_images, self.data.data["val_pars"][:, self.parameter]),
								  callbacks=[utils.callback_checkpoint(self.settings[self.parameter]["weights"]),
								  			 utils.callback_early_stop("val_mean_absolute_error", 
											   						   self.settings[self.parameter]["stop_size"], 
																	   self.settings[self.parameter]["stop_epochs"])])

		if self.output_name is not None:
			# Save history
			utils.save_regression_history(train_history, self.output_name)

			# Evaluate the test sample on the trained model and save output to file
			test_output = model.predict(test_images, verbose=0)
			utils.save_regression_output(self.data.data["test_labels"], 
										 self.data.data["test_pars"], 
										 test_output, self.output_name)

	def evaluate(self):
		print("PPENetwork: evaluate() ...")

	def optimise(self):
		print("PPENetwork: optimise() ...")

		x, val_images, test_images = self.data.get_images(self.settings[self.parameter]["norm"])		
		y = self.data.data["train_labels"]	

		p = {"learning_rate":[0.001], "batch_size":[500], "epochs":[40],
			 "filters_1":[32], "size_1":[3], "pool_1":[2],
			 "filters_2":[64], "size_2":[3], "pool_2":[2],
			 "dense":[128], "dropout":[0.0], "stop_size":[5], "stop_epochs":[5]}

		t = ta.Scan(x=x, y=y, model=models.ppe_model_fit, grid_downsample=1.0, 
					params=p, dataset_name='ppe_model', experiment_no='0',
					clear_tf_session=False, val_split=0.2)

class PARNetwork(Network): # Combined particle parameter estimation network
	def __init__(self, data_handler, output_name, settings):
		Network.__init__(self, data_handler, output_name, settings)

	def train(self):
		print("PARNetwork: train() ...")
		print("Output:{0} Weights:{1}".format(self.output_name, self.settings["weights"])) 
		print("Norm:{0} BatchSize:{1} LearningRate:{2} NumEpochs:{3}".format(
			  self.settings["norm"], self.settings["batch_size"],
			  self.settings["learning_rate"], self.settings["epochs"]))

		# Get the specific Keras parameter model
		model = models.par_model(self.data.get_image_shape(), self.settings["learning_rate"])

		# Get the images and pars from the different samples
		train_images, val_images, test_images = self.data.get_images(self.settings["norm"])			
		train_pars, val_pars, test_pars = self.data.get_transformed_pars()

		# Fit the model with the training data and store the training "history"
		train_history = model.fit(train_images, 
								  train_pars,
								  batch_size=int(self.settings["batch_size"]), 
								  epochs=int(self.settings["epochs"]), 
								  verbose=1,
								  validation_data=(val_images, val_pars),
								  callbacks=[utils.callback_checkpoint(self.settings["weights"]),
								  			 utils.callback_early_stop("val_mean_absolute_error", 
											   						   self.settings["stop_size"], 
																	   self.settings["stop_epochs"])])

		if self.output_name is not None:
			# Save history
			utils.save_regression_history(train_history, self.output_name)

			# Evaluate the test sample on the trained model and save output to file
			test_output = model.predict(test_images, verbose=0)
			test_output = self.data.inverse_transform_pars(test_output)
			utils.save_regression_output(self.data.data["test_labels"], 
										 self.data.data["test_pars"], 
										 test_output, self.output_name)

# Parse the command line argument options
def parse_args():
	parser = argparse.ArgumentParser(description='Train/Evaluate/Optimise CHIPS CVN Networks...')

	# Input and output files (history file will have similar name to output file)
	parser.add_argument('input', help = 'Path to combined input "image" .txt file')
	parser.add_argument('-o', '--output',   default = '../../output/output.txt', help = 'Output .txt file')
	parser.add_argument('-t', '--type',	 	default = 'ppe', help = '(ppe, pid, par)')
	parser.add_argument('-w', '--weights', 	default = '../../output/cp.ckpt',	 help = 'Network weights')

	# What function are we doing?
	parser.add_argument('--train', 		action = 'store_true',  help = 'Use to train the network')
	parser.add_argument('--eval',  		action = 'store_true',  help = 'Use to evaluate on the network')
	parser.add_argument('--opt', 		action = 'store_true', 	help = 'Use to optimise the network')

	# Network Parameters
	parser.add_argument('-p', '--parameter',	default = -1,	help = 'Parameter to fit (lepton Energy = 6)')
	parser.add_argument('--val_frac',   		default = 0.1,  help = 'Fraction of events for validation (0.1)')
	parser.add_argument('--test_frac',  		default = 0.1,  help = 'Fraction of events for testing (0.1)')
	parser.add_argument('--no_hit',  	action = 'store_true', 	help = 'Do not use hit channel')
	parser.add_argument('--no_time', 	action = 'store_true', 	help = 'Do not use time channel')
	parser.add_argument('-s', '--image_size', 	default = 32, 	help = 'Input image size (32)')

	# Check arguments
	args = parser.parse_args()

	if args.type not in ["ppe", "pid", "par"]:
		print("Error: Specify Network!")
		sys.exit()			
	elif args.no_hit and args.no_time:
		print("Error: Need to use at least one channel!")
		sys.exit()	
	elif args.train and args.eval or args.train and args.opt or args.eval and args.opt:
		print("Error: Can only do one thing at a time!")
		sys.exit()
	elif args.train and args.eval and args.opt:
		print("Error: Can only do one thing at a time!")
		sys.exit()	
	elif not args.train and not args.eval and not args.opt:
		print("Error: Need to specify something to do!")
		sys.exit()				

	return args  

def main():
	args = parse_args() # Get the command line arguments

	# Load the data into the DataHandler and print the summary
	data = utils.DataHandler(args.input, float(args.val_frac), float(args.test_frac),
					   		 int(args.image_size), args.no_hit, args.no_time)
	data.load_data()
	data.print()

	if args.type == "ppe":

		# Load settings into dict
		settings_file = open("ppe_settings.dat", "r")
		settings_dict = csv.DictReader(settings_file)
		settings_list = []
		for par in settings_dict:
			settings_list.append(par)
		settings_file.close()

		network = PPENetwork(data, args.output, settings_list, int(args.parameter))
		if args.train:
			network.train()
		elif args.eval:
			network.evaluate()
		elif args.opt:
			network.optimise()
		
	elif args.type == "pid":

		# Load settings into dict
		settings_file = open("pid_settings.dat", "r")
		settings_dict = csv.DictReader(settings_file)
		settings_list = []
		for par in settings_dict:
			settings_list.append(par)
		settings_file.close()

		network = PIDNetwork(data, args.output, settings_list[0])
		if args.train:
			network.train()
		elif args.eval:
			network.evaluate()
		elif args.opt:
			network.optimise()	

	elif args.type == "par":

		# Load settings into dict
		settings_file = open("par_settings.dat", "r")
		settings_dict = csv.DictReader(settings_file)
		settings_list = []
		for par in settings_dict:
			settings_list.append(par)
		settings_file.close()

		network = PARNetwork(data, args.output, settings_list[0])
		if args.train:
			network.train()
		elif args.eval:
			network.evaluate()
		elif args.opt:
			network.optimise()


if __name__=='__main__':
	main()