# data methods, handles the IO from map and tfrecord files
# Author: Josh Tingey
# Email: j.tingey.16@ucl.ac.uk

import os
import argparse
from joblib import Parallel, delayed
import multiprocessing

import uproot
import numpy as np
import pandas as pd
import tensorflow as tf

import config
import utils

def parse(serialized_example):
    features = {
        'category': tf.io.FixedLenFeature([], tf.string),
        'parameters': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, features)

    image = tf.io.decode_raw(example['image'], tf.float32)
    image = tf.reshape(image, [64, 64, 2])
    pars = tf.io.decode_raw(example['parameters'], tf.float32)

    return image, pars[6]

def dataset(dirs, par=0):
    files = []  # Add all files in dirs to a list
    for d in dirs:
        for file in os.listdir(d):
            files.append(os.path.join(d, file))

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(tf.data.TFRecordDataset, cycle_length=len(files), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(64, drop_remainder=True)
    
    return ds

# Returns a bytes_list from a string / byte
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Returns a float_list from a float / double
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# Returns an int64_list from a bool / enum / int / uint
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_examples(true, reco):
    # Get the numpy arrays from the .root map file
    categories = true.array("true_category")
    parameters = np.stack((true.array("true_vtx_x"), true.array("true_vtx_y"), true.array("true_vtx_z"),
                           true.array("true_dir_costheta"), true.array("true_dir_phi"), true.array("true_nu_energy"),
                           true.array("true_lep_energy")), axis=1)
    images = np.stack((reco.array("raw_hit_map_origin"), reco.array("raw_time_map_origin")), axis=3)

    # Generate examples using feature map
    examples = []
    for i in range(len(categories)):
        feature = {
            'category': _bytes_feature(categories[i].tostring()),
            'parameters': _bytes_feature(parameters[i].tostring()),
            'image': _bytes_feature(images[i].tostring())
        }
        examples.append(tf.train.Example(features=tf.train.Features(feature=feature)))

    return examples

def write_examples(name, examples):
    with tf.io.TFRecordWriter(name) as writer:      
        writer.write(example.SerializeToString())

def preprocess_file(conf, file, out_dir):
    # Add all examples from the file to a list
    examples = []
    file_u = uproot.open(file)
    try: # Catch when a file is bad
        examples = get_examples(file_u["true"], file_u["reco"])
    except:
        return

    # Split into training, validation and testing samples
    val_split = int((1.0-config.test_split-config.val_split) * len(examples)) 
    test_split = int((1.0-config.test_split) * len(examples)) 
    train_examples = examples[:val_split]
    val_examples = examples[val_split:test_split]
    test_examples = examples[test_split:]

    name, ext = os.path.splitext(file)
    base = os.path.basename(name)
    write_examples(os.path.join(out_dir, "train/", base + "_train.tfrecords"), train_examples)
    write_examples(os.path.join(out_dir, "val/", base + "_val.tfrecords"), val_examples)
    write_examples(os.path.join(out_dir, "test/", base + "_test.tfrecords"), test_examples)

def preprocess_dir(conf, in_dir, out_dir):
    # We can parallelise the process as each file is independent
    Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(preprocess_file)(conf, os.path.join(in_dir, f), out_dir) for f in os.listdir(in_dir))

def parse_args():
    parser = argparse.ArgumentParser( description='CHIPS CVN Data')
    parser.add_argument('-i', '--in_dir', help='path to input directory')
    parser.add_argument('-o', '--out_dir', help='path to output directory')
    parser.add_argument('-c', '--conf', help='Config .json file', default='config/preprocessing.json')
    return parser.parse_args()

def main():
    args = parse_args()
    conf = config.get_config(args.conf)
    preprocess_dir(conf, args.in_dir, args.out_dir)

if __name__ == '__main__':
    main()