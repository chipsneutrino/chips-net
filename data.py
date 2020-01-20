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

import uproot
import numpy as np
import tensorflow as tf

import config


def parse(serialised_example, shape):
    """Parses a single serialised example an image and labels dict."""
    features = {
        'category': tf.io.FixedLenFeature([], tf.string),
        'parameters': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(serialised_example, features)

    category = tf.io.decode_raw(example['category'], tf.int32)
    parameters = tf.io.decode_raw(example['parameters'], tf.float32)
    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, shape)

    labels = {  # We generate a dictionary with all the true labels
        'category': category,
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

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=len(files),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.map(lambda x: parse(x, shape), num_parallel_calls=tf.data.experimental.AUTOTUNE)
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


def gen_examples(true, reco, conf):
    """Generates a list of examples from the input .root map file."""
    # Get the numpy arrays from the .root map file
    pdgs = true.array("true_pdg")
    types = true.array("true_type")
    categories = get_categories(pdgs, types, conf)
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
    for i in range(len(categories)):
        feature_dict = {
            'category': bytes_feature(categories[i].tostring()),
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


def preprocess_file(file, out_dir, split, conf):
    """Preprocess a .root map file into train, val and test tfrecords files."""
    examples = []
    file_u = uproot.open(file)
    try:
        examples = gen_examples(file_u["true"], file_u["reco"], conf)
    except Exception as err:  # Catch when there is an uproot exception
        print("Error:", type(err), err)
        return

    # Split into training, validation and testing samples
    val_split = int((1.0-split-split) * len(examples))
    test_split = int((1.0-split) * len(examples))
    train_examples = examples[:val_split]
    val_examples = examples[val_split:test_split]
    test_examples = examples[test_split:]

    name, ext = os.path.splitext(file)
    base = os.path.basename(name)
    write_examples(os.path.join(out_dir, "train/", base + "_train.tfrecords"), train_examples)
    write_examples(os.path.join(out_dir, "val/", base + "_val.tfrecords"), val_examples)
    write_examples(os.path.join(out_dir, "test/", base + "_test.tfrecords"), test_examples)


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


def preprocess_dir(in_dir, out_dir, split, conf, single):
    """Preprocess all the files from the input directory into tfrecords."""
    if not single:  # File independence allows for parallelisation
        Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(
            preprocess_file)(os.path.join(in_dir, f), out_dir,
                             split, conf) for f in os.listdir(in_dir))
    else:  # For debugging we keep the option to use a single process
        for f in os.listdir(in_dir):
            preprocess_file(os.path.join(in_dir, f), out_dir, split, conf)


def parse_args():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='CHIPS CVN Data')
    parser.add_argument('directory', help='path to input directory')
    parser.add_argument('-s', '--split', help='val and test split fraction', default=0.1)
    parser.add_argument('-g', '--geom', help='detector geometry name')
    parser.add_argument('--single', action='store_true', help='Use a single process')
    return parser.parse_args()


def main():
    """Main function called by script."""
    args = parse_args()
    conf = config.get_config("config/categories.json")
    in_dir, out_dir = make_directories(args.directory, args.geom)
    preprocess_dir(in_dir, out_dir, args.split, conf, args.single)


if __name__ == '__main__':
    main()
