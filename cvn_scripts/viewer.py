# .txt image file viewer
#
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk
#

import os.path
import os.path as path
import sys
import argparse
import os

import models as models
import utils as utils

def parse_args():
	parser = argparse.ArgumentParser(description='Evaluate/Train CHIPS CVN PID Network')
	parser.add_argument('file', help = 'Path to input "image" .txt file')
	parser.add_argument('-s', '--imageSize', default = 32, help = 'Input image size (32)')
	return parser.parse_args()	  

def main():
	args = parse_args() # Get the command line arguments
	data = utils.DataHandler(args.file, 0.0, 0.0, int(args.imageSize), False, False)
	data.load_data()
	data.print()

	for event in range(100):
		data.plot_image(event)

if __name__=='__main__':
	main()
