"""Data methods and preprocessing functionality

Author: Josh Tingey
Email: j.tingey.16@ucl.ac.uk

This script allows for the preprocessing of .root hit map files into
tfrecords files in a parallelisable fashion. It can also be imported
as a module to generate tensorflow datasets from the tfrecords files.
"""

import os
import argparse
from joblib import Parallel, delayed
import multiprocessing
import random

import uproot
import numpy as np
import tensorflow as tf


def parse(serialised_example, shape, pdg_table, type_table):
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
    image = tf.reshape(image, shape)

    labels = {  # We generate a dictionary with all the true labels
        'pdg': pdg_table.lookup(types[0]),
        'type': type_table.lookup(types[1]),
        'vtxX': parameters[0],
        'vtxY': parameters[1],
        'vtxZ': parameters[2],
        'dirTheta': parameters[3],
        'dirPhi': parameters[4],
        'nuEnergy': parameters[5],
        'lepEnergy': parameters[6],
    }

    return image, labels


def dataset(dirs, shape):
    """Returns a dataset formed from all the files in the input directories."""
    files = []  # Add all files in dirs to a list
    for d in dirs:
        for file in os.listdir(d):
            files.append(os.path.join(d, file))

    random.shuffle(files)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=len(files),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # PDG classification mapping
    pdg_keys = tf.constant([12, 14])
    pdg_vals = tf.constant([0,  1])
    pdg_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(pdg_keys, pdg_vals), -1)

    # Event type classification mapping
    type_keys = tf.constant([0, 98, 98, 2, 6, 7, 8, 9, 92, 96, 1, 3, 4, 5, 91, 97, 100])
    type_vals = tf.constant([0,  0,  0, 1, 1, 1, 1, 1,  1,  1, 2, 3, 3, 3,  4,  5,   6])
    type_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(type_keys, type_vals), -1)

    ds = ds.map(lambda x: parse(x, shape, pdg_table, type_table),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ds


def get_train_and_val_ds(dirs, shape):
    """Returns the train and val datasets from the input directories."""
    train_dirs = []
    val_dirs = []
    for directory in dirs:
        train_dirs.append(os.path.join(directory, "train"))
        val_dirs.append(os.path.join(directory, "val"))

    train_ds = dataset(train_dirs, shape)
    val_ds = dataset(val_dirs, shape)
    return train_ds, val_ds


def get_test_ds(dirs, shape):
    """Returns the test dataset from the input directories."""
    test_dirs = []
    for directory in dirs:
        test_dirs.append(os.path.join(directory, "test"))

    test_ds = dataset(test_dirs, shape)
    return test_ds


def get_categories(pdgs, types, conf):
    """assigns the correct category given the true_pdg and true_type."""
    categories = np.zeros((len(pdgs)), dtype=np.int32)
    for event in range(len(pdgs)):
        key = str(pdgs[event]) + "-" + str(types[event])
        categories[event] = conf[key]
    return categories


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def gen_examples(true, reco):
    """Generates a list of examples from the input .root map file."""
    # Get the numpy arrays from the .root map file
    types = np.stack((true.array("true_pdg"),
                      true.array("true_type")), axis=1)
    parameters = np.stack((true.array("true_vtx_x"),
                           true.array("true_vtx_y"),
                           true.array("true_vtx_z"),
                           true.array("true_dir_costheta"),
                           true.array("true_dir_phi"),
                           true.array("true_nu_energy"),
                           true.array("true_lep_energy")), axis=1)
    images = np.stack((reco.array("filtered_charge_map_origin"),
                       reco.array("filtered_time_map_vtx"),
                       reco.array("filtered_hit_hough_map_vtx")), axis=3)

    examples = []  # Generate examples using a feature dict
    for i in range(len(types)):
        feature_dict = {
            'types': bytes_feature(types[i].tostring()),
            'parameters': bytes_feature(parameters[i].tostring()),
            'image': bytes_feature(images[i].tostring())
        }
        examples.append(tf.train.Example(features=tf.train.Features(feature=feature_dict)))

    return examples


def write_examples(name, examples):
    """Write a list of TF Examples to a tfrecords file."""
    with tf.io.TFRecordWriter(name) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def preprocess_file(num, files, out_dir, split):
    """Preprocess a .root map file into train, val and test tfrecords files."""
    print("Processing job {}...".format(num))
    examples = []
    for file in files:
        file_u = uproot.open(file)
        try:
            examples.extend(gen_examples(file_u["true"], file_u["reco"]))
        except Exception as err:  # Catch when there is an uproot exception
            print("Error:", type(err), err)
            pass

    # Split into training, validation and testing samples
    val_split = int((1.0-split-split) * len(examples))
    test_split = int((1.0-split) * len(examples))
    train_examples = examples[:val_split]
    val_examples = examples[val_split:test_split]
    test_examples = examples[test_split:]

    write_examples(os.path.join(out_dir, "train/", str(num) + "_train.tfrecords"), train_examples)
    write_examples(os.path.join(out_dir, "val/", str(num) + "_val.tfrecords"), val_examples)
    write_examples(os.path.join(out_dir, "test/", str(num) + "_test.tfrecords"), test_examples)


def make_directories(directory, geom):
    """Makes the output directory structure."""
    in_dir = os.path.join(directory, "map/", geom)
    out_dir = os.path.join(directory, "tf/", geom)
    try:
        os.mkdir(out_dir)
        os.mkdir(os.path.join(out_dir, "train/"))
        os.mkdir(os.path.join(out_dir, "val/"))
        os.mkdir(os.path.join(out_dir, "test/"))
    except FileExistsError:
        pass
    return in_dir, out_dir


def preprocess_dir(in_dir, out_dir, split, join, single):
    """Preprocess all the files from the input directory into tfrecords."""
    files = [os.path.join(in_dir, file) for file in os.listdir(in_dir)]
    file_lists = [files[n:n+join] for n in range(0, len(files), join)]
    if not single:  # File independence allows for parallelisation
        Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
            preprocess_file)(counter, f_list, out_dir,
                             split) for counter, f_list in enumerate(file_lists))
    else:  # For debugging we keep the option to use a single process
        for counter, f_list in enumerate(file_lists):
            preprocess_file(counter, f_list, out_dir, split)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN Data')
    parser.add_argument('directory', help='path to input directory')
    parser.add_argument('-s', '--split', help='val and test split fraction', default=0.1)
    parser.add_argument('-g', '--geom', help='detector geometry name', default="chips_1200_sk1pe")
    parser.add_argument('-j', '--join', help='how many input files to join together', default=10)
    parser.add_argument('--single', action='store_true', help='Use a single process')
    return parser.parse_args()


def main():
    """Main function called by script."""
    args = parse_args()
    in_dir, out_dir = make_directories(args.directory, args.geom)
    preprocess_dir(in_dir, out_dir, args.split, args.join, args.single)


if __name__ == '__main__':
    main()
