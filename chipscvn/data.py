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
    """Generates tf datasets from the configuration."""
    def __init__(self, config):
        self.config = config
        self.init()

    def init(self):
        """Initialise the PDG, type and category lookup tables and input directories."""
        pdg_keys = tf.constant([11, 12, 13, 14])
        pdg_vals = tf.constant([0,  0,  1,  1])
        self.pdg_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(pdg_keys, pdg_vals), -1)

        type_keys = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 91, 92, 96, 97, 98, 99, 100])
        type_vals = tf.constant([6, 0, 4, 1, 1, 1, 4, 4, 4, 4,  2,  4,  4,  3,  6,  6,   5])
        self.type_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(type_keys, type_vals), -1)

        # Category keys are a string of pdg+type, e.g an nuel ccqe event is '0'+'0' = '00'
        cat_keys = tf.constant(['00', '10', '01', '11', '02', '12', '03', '13', '04', '14', '06', '16', '15'], dtype=tf.string)
        cat_vals = tf.constant([  0,    1,    2,    3,    4,    5,    6,    7,    8,    8,    8,    8,    9])
        self.cat_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(cat_keys, cat_vals), -1)

        # Generate the lists of train, val and test file directories from the configuration
        self.train_dirs = [os.path.join(in_dir, 'train') for in_dir in self.config.data.input_dirs]
        self.val_dirs = [os.path.join(in_dir, 'val') for in_dir in self.config.data.input_dirs]
        self.test_dirs = [os.path.join(in_dir, 'test') for in_dir in self.config.data.input_dirs]

    def parse(self, serialised_example):
        """Parses a single serialised example into an input plus a labels dict."""
        features = {
            'true_pars_i': tf.io.FixedLenFeature([], tf.string),
            'true_pars_f': tf.io.FixedLenFeature([], tf.string),
            'reco_pars_i': tf.io.FixedLenFeature([], tf.string),
            'reco_pars_f': tf.io.FixedLenFeature([], tf.string),
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialised_example, features)

        # Decode the parameter arrays using their types
        true_pars_i = tf.io.decode_raw(example['true_pars_i'], tf.int32)
        true_pars_f = tf.io.decode_raw(example['true_pars_f'], tf.float32)
        reco_pars_i = tf.io.decode_raw(example['reco_pars_i'], tf.int32)
        reco_pars_f = tf.io.decode_raw(example['reco_pars_f'], tf.float32)

        # Unpack the pdg and type and use to determine event category
        pdg = self.pdg_table.lookup(true_pars_i[0])
        type = self.type_table.lookup(true_pars_i[1])
        category = self.cat_table.lookup(tf.strings.join((tf.strings.as_string(pdg), 
                                                          tf.strings.as_string(type))))

        labels = {  # We generate a dictionary with all the true labels
            'true_pdg': pdg,
            'true_type': type,
            'true_category': category,
            'true_vtxX': true_pars_f[0],
            'true_vtxY': true_pars_f[1],
            'true_vtxZ': true_pars_f[2],
            'true_dirTheta': true_pars_f[3],
            'true_dirPhi': true_pars_f[4],
            'true_nuEnergy': true_pars_f[5],
            'true_lepEnergy': true_pars_f[6],
        }

        # Decode and reshape the "image" arrays into a tf tensor
        image = tf.io.decode_raw(example['image'], tf.float32)
        image = tf.reshape(image, self.config.data.img_shape)

        # Split the image into charge+time and hough tensors
        ct_image, h_image = tf.split(image, [2, 1], 2)
        tf.expand_dims(h_image, 2) 

        # Apply augmentation to charge+time image
        charge_rand = tf.random.normal(shape=[64, 64], mean=0, stddev=self.config.data.charge_spread)
        time_rand = tf.random.normal(shape=[64, 64], mean=0, stddev=self.config.data.time_spread)
        tot_rand = tf.stack([charge_rand, time_rand], axis=2)
        ct_image = tf.math.add(ct_image, tot_rand)

        # Apply augmentation to hough image
        hough_rand = tf.random.normal(shape=[64, 64, 1], mean=0, stddev=self.config.data.hough_spread)
        h_image = tf.math.add(h_image, hough_rand)

        # Calculate cut booleans
        pi_on_180 = 0.017453292519943295
        z_cut = tf.math.less_equal(tf.math.abs(reco_pars_f[6]), self.config.data.cuts.fiducial_z)
        r_cut = tf.math.less_equal(
            tf.math.sqrt(tf.math.pow(reco_pars_f[4], 2) + tf.math.pow(reco_pars_f[5], 2)),
            self.config.data.cuts.fiducial_r
        )
        dir_cut = tf.math.greater_equal(tf.math.multiply(
            tf.math.sin(tf.math.acos(reco_pars_f[7])), tf.math.cos(reco_pars_f[8] * pi_on_180)),
            self.config.data.cuts.dir_x
        )
        q_cut = tf.math.greater_equal(reco_pars_f[0], self.config.data.cuts.total_q)
        
        inputs = {  # We generate a dictionary with the images and other input parameters
            'ct_image': ct_image,
            'h_image': h_image,
            'raw_num_hits': reco_pars_i[0],
            'filtered_num_hits': reco_pars_i[1],
            'num_hough_rings': reco_pars_i[2],
            'raw_total_digi_q': reco_pars_f[0],
            'filtered_total_digi_q': reco_pars_f[1],
            'first_ring_height': reco_pars_f[2],
            'last_ring_height': reco_pars_f[3],
            'reco_vtxX': reco_pars_f[4],
            'reco_vtxY': reco_pars_f[5],
            'reco_vtxZ': reco_pars_f[6],
            'reco_dirTheta': reco_pars_f[7],
            'reco_dirPhi': reco_pars_f[8],
            'z_cut': z_cut,
            'r_cut': r_cut,
            'dir_cut': dir_cut,
            'q_cut': q_cut
        }

        return inputs, labels

    def fiducial_cut(self, inputs, labels):
        """Applies a fiducial volume cut on the reco vtx location."""
        return (inputs["z_cut"] and inputs["r_cut"])

    def dir_cut(self, inputs, labels):
        """Applies a beam direction cut on the reco direction."""
        return inputs['dir_cut']

    def charge_cut(self, inputs, labels):
        """Applies a total event charge cut."""
        return inputs['q_cut']

    def dataset(self, dirs):
        """Returns a dataset formed from all the files in the input directories."""
        files = []  # Add all files in dirs to a list
        for d in dirs:
            for file in os.listdir(d):
                files.append(os.path.join(d, file))

        random.shuffle(files)  # We shuffle the list as an additionally randomisation to "interleave"
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.interleave(tf.data.TFRecordDataset,
                           cycle_length=len(files),
                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda x: self.parse(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    def train_data(self):
        """Returns the training dataset."""
        ds = self.dataset(self.train_dirs)
        ds = ds.filter(self.dir_cut)
        ds = ds.filter(self.charge_cut)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(self.config.data.max_examples)  # Only take 10% of max examples
        return ds

    def val_data(self):
        """Returns the validation dataset."""
        ds = self.dataset(self.val_dirs)
        ds = ds.filter(self.dir_cut)
        ds = ds.filter(self.charge_cut)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(int(self.config.data.max_examples*0.1))  # Only take 10% of max examples
        return ds

    def test_data(self):
        """Returns the testing dataset."""
        ds = self.dataset(self.test_dirs)
        ds = ds.batch(self.config.data.batch_size, drop_remainder=True)
        ds = ds.take(int(self.config.data.max_examples*0.1))  # Only take 10% of max examples
        return ds


class DataCreator:
    """Generates tfrecord files from ROOT map files."""
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

        # Get the numpy arrays from the .root map file, we need to seperate by type
        # for the deserialisation during reading to work correctly.
        true_pars_i = np.stack((  # True Parameters (integers)
            true.array('true_pdg'),
            true.array('true_type')),
            axis=1)
        true_pars_f = np.stack((  # True Parameters (floats)
            true.array('true_vtx_x'),
            true.array('true_vtx_y'),
            true.array('true_vtx_z'),
            true.array('true_dir_costheta'),
            true.array('true_dir_phi'),
            true.array('true_nu_energy'),
            true.array('true_lep_energy')),
            axis=1)
        reco_pars_i = np.stack((  # Reco Parameters (integers)
            reco.array('raw_num_hits'),
            reco.array('filtered_num_hits'),
            reco.array('num_hough_rings')),
            axis=1)
        reco_pars_f = np.stack((  # Reco Parameters (floats)
            reco.array('raw_total_digi_q'),
            reco.array('filtered_total_digi_q'),
            reco.array('first_ring_height'),
            reco.array('last_ring_height'),
            reco.array('vtx_x'),
            reco.array('vtx_y'),
            reco.array('vtx_z'),
            reco.array('dir_theta'),
            reco.array('dir_phi')),
            axis=1)
        image = np.stack((  # Image
            reco.array('filtered_charge_map_vtx'),
            reco.array('filtered_time_map_vtx'),
            reco.array('filtered_hit_hough_map_vtx')),
            axis=3)

        examples = []  # Generate examples using a feature dict
        for i in range(len(true_pars_i)):
            feature_dict = {
                'true_pars_i': self.bytes_feature(true_pars_i[i].tostring()),
                'true_pars_f': self.bytes_feature(true_pars_f[i].tostring()),
                'reco_pars_i': self.bytes_feature(reco_pars_i[i].tostring()),
                'reco_pars_f': self.bytes_feature(reco_pars_f[i].tostring()),
                'image': self.bytes_feature(image[i].tostring())
            }
            examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))

        return examples

    def write_examples(self, name, examples):
        """Write a list of examples to a tfrecords file."""
        with tf.io.TFRecordWriter(name) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def preprocess_file(self, num, files):
        """Preprocess joined .root map files into train, val and test tfrecords files."""
        print('Processing job {}...'.format(num))
        examples = []
        for file in files:
            file_u = uproot.open(file)
            try:
                examples.extend(self.gen_examples(file_u['true'], file_u['reco']))
            except Exception as err:  # Catch when there is an uproot exception and skip
                print('Error:', type(err), err)
                pass

        # Split into training, validation and testing samples
        val_split = int((1.0-self.split-self.split) * len(examples))
        test_split = int((1.0-self.split) * len(examples))
        train_examples = examples[:val_split]
        val_examples = examples[val_split:test_split]
        test_examples = examples[test_split:]

        self.write_examples(
            os.path.join(self.out_dir, 'train/', str(num) + '_train.tfrecords'), train_examples)
        self.write_examples(
            os.path.join(self.out_dir, 'val/', str(num) + '_val.tfrecords'), val_examples)
        self.write_examples(
            os.path.join(self.out_dir, 'test/', str(num) + '_test.tfrecords'), test_examples)

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
