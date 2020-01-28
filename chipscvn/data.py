"""Data creation and loading module

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This module contains both the DataCreator and DataLoader classes, these
are used to firstly generate tfrecords files from ROOT hitmap files and
then to read these on the fly using tf.datasets at model training or
evaluation.
"""

import os
from joblib import Parallel, delayed
import multiprocessing
import random

import uproot
import numpy as np
import tensorflow as tf


class DataLoader:
    """Data loader that generates datasets from the configuration."""
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        """Initialise the PDG and type lookup tables and input directories."""
        pdg_keys = tf.constant([12, 14])
        pdg_vals = tf.constant([0,  1])
        self.pdg_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(pdg_keys, pdg_vals), -1)

        type_keys = tf.constant([0, 98, 98, 2, 6, 7, 8, 9, 92, 96, 1, 3, 4, 5, 91, 97, 100])
        type_vals = tf.constant([0,  0,  0, 1, 1, 1, 1, 1,  1,  1, 2, 3, 3, 3,  4,  5,   6])
        self.type_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(type_keys, type_vals), -1)

        self.train_dirs = [os.path.join(in_dir, "train") for in_dir in self.config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, "val") for in_dir in self.config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, "test") for in_dir in self.config.data.input_dirs]

    def parse(self, serialised_example):
        """Parses a single serialised example an image and labels dict."""
        features = {
            'types': tf.io.FixedLenFeature([], tf.string),
            'parameters': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialised_example, features)

        types = tf.io.decode_raw(example['types'], tf.int32)
        parameters = tf.io.decode_raw(example['parameters'], tf.float32)
        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, self.config.data.img_shape)

        labels = {  # We generate a dictionary with all the true labels
            'pdg': self.pdg_table.lookup(types[0]),
            'type': self.type_table.lookup(types[1]),
            'vtxX': parameters[0],
            'vtxY': parameters[1],
            'vtxZ': parameters[2],
            'dirTheta': parameters[3],
            'dirPhi': parameters[4],
            'nuEnergy': parameters[5],
            'lepEnergy': parameters[6],
        }

        return image, labels

    def dataset(self, dirs):
        """Returns a dataset formed from all the files in the input directories."""
        files = []  # Add all files in dirs to a list
        for d in dirs:
            for file in os.listdir(d):
                files.append(os.path.join(d, file))

        random.shuffle(files)
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.interleave(tf.data.TFRecordDataset,
                           cycle_length=len(files),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: self.parse(x),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(self.config.data.max_examples)
        
        return ds

    def train_data(self):
        """Returns the training dataset."""
        return self.dataset(self.train_dirs)

    def val_data(self):
        """Returns the validation dataset."""
        return self.dataset(self.val_dirs)

    def test_data(self):
        """Returns the testing dataset."""
        return self.dataset(self.test_dirs)


class DataCreator:
    """Data creator that generates tfrecord files from ROOT map files."""
    def __init__(self, directory, geom, split, join, single):
        self.split = split
        self.join = join
        self.single = single
        self.init(directory, geom)

    def init(self, directory, geom):
        """Initialise the output directories."""
        self.in_dir = os.path.join(directory, "map/", geom)
        self.out_dir = os.path.join(directory, "tf/", geom)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "train/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "val/"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "test/"), exist_ok=True)

    def bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def gen_examples(self, true, reco):
        """Generates a list of examples from the input .root map file."""
        # Get the numpy arrays from the .root map file
        types = np.stack((  # Integer labels
            true.array("true_pdg"),
            true.array("true_type")),
            axis=1)
        parameters = np.stack((  # Float labels
            true.array("true_vtx_x"),
            true.array("true_vtx_y"),
            true.array("true_vtx_z"),
            true.array("true_dir_costheta"),
            true.array("true_dir_phi"),
            true.array("true_nu_energy"),
            true.array("true_lep_energy")),
            axis=1)
        images = np.stack((
            reco.array("filtered_charge_map_origin"),
            reco.array("filtered_time_map_vtx"),
            reco.array("filtered_hit_hough_map_vtx")),
            axis=3)

        examples = []  # Generate examples using a feature dict
        for i in range(len(types)):
            feature_dict = {
                'types': self.bytes_feature(types[i].tostring()),
                'parameters': self.bytes_feature(parameters[i].tostring()),
                'image': self.bytes_feature(images[i].tostring())
            }
            examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))

        return examples

    def write_examples(self, name, examples):
        """Write a list of tf.train.Example to a tfrecords file."""
        with tf.io.TFRecordWriter(name) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def preprocess_file(self, num, files):
        """Preprocess joined .root map files into train, val and test tfrecords files."""
        print("Processing job {}...".format(num))
        examples = []
        for file in files:
            file_u = uproot.open(file)
            try:
                examples.extend(self.gen_examples(file_u["true"], file_u["reco"]))
            except Exception as err:  # Catch when there is an uproot exception and skip
                print("Error:", type(err), err)
                pass

        # Split into training, validation and testing samples
        val_split = int((1.0-self.split-self.split) * len(examples))
        test_split = int((1.0-self.split) * len(examples))
        train_examples = examples[:val_split]
        val_examples = examples[val_split:test_split]
        test_examples = examples[test_split:]

        self.write_examples(
            os.path.join(self.out_dir, "train/", str(num) + "_train.tfrecords"), train_examples)
        self.write_examples(
            os.path.join(self.out_dir, "val/", str(num) + "_val.tfrecords"), val_examples)
        self.write_examples(
            os.path.join(self.out_dir, "test/", str(num) + "_test.tfrecords"), test_examples)

    def preprocess(self):
        """Preprocess all the files from the input directory into tfrecords."""
        files = [os.path.join(self.in_dir, file) for file in os.listdir(self.in_dir)]
        file_lists = [files[n:n+self.join] for n in range(0, len(files), self.join)]
        if not self.single:  # File independence allows for parallelisation
            Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
                self.preprocess_file)(counter, f_list) for counter, f_list in enumerate(file_lists))
        else:  # For debugging we keep the option to use a single process
            for counter, f_list in enumerate(file_lists):
                self.preprocess_file(counter, f_list)
