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

        type_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 91, 92, 96, 97, 98, 99, 100])
        type_vals = tf.constant([6, 0, 4, 1, 1, 1, 4, 4, 4, 4,  2,  4,  4,  3,  6,  6,   5])
        self.type_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(type_keys, type_vals), -1)

        cat_keys = tf.constant(["00", "10", "01", "11", "02", "12", "03", "13", "04", "14", "06", "16", "15"], dtype=tf.string)
        cat_vals = tf.constant([  0,    1,    2,    3,    4,    5,    6,    7,    8,    8,    8,    8,    9])
        self.cat_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(cat_keys, cat_vals), -1)

        self.train_dirs = [os.path.join(in_dir, "train") for in_dir in self.config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, "val") for in_dir in self.config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, "test") for in_dir in self.config.data.input_dirs]

    def parse(self, serialised_example):
        """Parses a single serialised example an image and labels dict."""
        features = {
            'types': tf.io.FixedLenFeature([], tf.string),
            'true_pars': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
            'reco_pars': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(serialised_example, features)

        # Get the true parameters and form the 'labels' dictionary
        types = tf.io.decode_raw(example['types'], tf.int32)
        true_pars = tf.io.decode_raw(example['true_pars'], tf.float32)

        pdg = self.pdg_table.lookup(types[0])
        type = self.type_table.lookup(types[1])
        category = self.cat_table.lookup(tf.strings.join((tf.strings.as_string(pdg), 
                                                          tf.strings.as_string(type))))

        labels = {  # We generate a dictionary with all the true labels
            'pdg': pdg,
            'type': type,
            'category': category,
            'vtxX': true_pars[0],
            'vtxY': true_pars[1],
            'vtxZ': true_pars[2],
            'dirTheta': true_pars[3],
            'dirPhi': true_pars[4],
            'nuEnergy': true_pars[5],
            'lepEnergy': true_pars[6],
        }

        # Get the images and reco parameters and form the 'inputs' dictionary
        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, self.config.data.img_shape)
        reco_pars = tf.io.decode_raw(example['reco_pars'], tf.float32)

        charge_rand = tf.random.normal(shape=[64, 64], mean=0, stddev=self.config.data.charge_spread)
        time_rand = tf.random.normal(shape=[64, 64], mean=0, stddev=self.config.data.time_spread)
        hough_rand = tf.random.normal(shape=[64, 64], mean=0, stddev=self.config.data.hough_spread)
        tot_rand = tf.stack([charge_rand, time_rand, hough_rand], axis=2)
        image = tf.math.add(image, tot_rand)

        inputs = {  # We generate a dictionary with the images and other reco parameters
            'image': image,
            'vtxX': reco_pars[0],
            'vtxY': reco_pars[1],
            'vtxZ': reco_pars[2],
            'dirTheta': reco_pars[3],
            'dirPhi': reco_pars[4],
        }

        return inputs, labels

    def dataset(self, dirs, example_fraction):
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
        ds = ds.take(int(self.config.data.max_examples*example_fraction))

        return ds

    def train_data(self):
        """Returns the training dataset."""
        return self.dataset(self.train_dirs, 1.0)

    def val_data(self):
        """Returns the validation dataset."""
        return self.dataset(self.val_dirs, 0.1)

    def test_data(self):
        """Returns the testing dataset."""
        return self.dataset(self.test_dirs, 0.1)


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
        types = np.stack((  # True Parameters (integers)
            true.array("true_pdg"),
            true.array("true_type")),
            axis=1)
        true_pars = np.stack((  # True Parameters (floats)
            true.array("true_vtx_x"),
            true.array("true_vtx_y"),
            true.array("true_vtx_z"),
            true.array("true_dir_costheta"),
            true.array("true_dir_phi"),
            true.array("true_nu_energy"),
            true.array("true_lep_energy")),
            axis=1)
        image = np.stack((  # Image
            reco.array("filtered_charge_map_vtx"),
            reco.array("filtered_time_map_vtx"),
            reco.array("filtered_hit_hough_map_vtx")),
            axis=3)
        reco_pars = np.stack((  # Reco Parameters
            reco.array("vtx_x"),
            reco.array("vtx_y"),
            reco.array("vtx_z"),
            reco.array("dir_theta"),
            reco.array("dir_phi")),
            axis=1)

        examples = []  # Generate examples using a feature dict
        for i in range(len(types)):
            feature_dict = {
                'types': self.bytes_feature(types[i].tostring()),
                'true_pars': self.bytes_feature(true_pars[i].tostring()),
                'image': self.bytes_feature(image[i].tostring()),
                'reco_pars': self.bytes_feature(reco_pars[i].tostring())
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
