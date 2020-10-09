import Config
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def serialize_example(seismic, label, height, width):
    """
    Creates a tf.train.Example message ready to be written to a file.
    :param seismic: Seismic image.
    :param label:   Label image.
    :param height:  Height of image.
    :param width:   Width of image.
    :return:        tf.train.Example message.
    """
    # Create dictionary mapping for each feature
    feature = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'seismic': tf.train.Feature(float_list=tf.train.FloatList(value=seismic.reshape(-1))),
        'label': tf.train.Feature(float_list=tf.train.FloatList(value=label.reshape(-1)))
    }

    # Create Features message using tf.train.Example
    exampleProto = tf.train.Example(features=tf.train.Features(feature=feature))

    return exampleProto.SerializeToString()


def write_dataset_to_tfrecord(seismic_data, labels, output_filepath):
    """
    Writes dataset into tfrecord file.
    :param seismic_data:        Numpy array containing seismic data of shape (inline, crossline, depth)
    :param labels:              Numpy array containing labels of shape (inline, crossline, depth)
    :param output_filepath:     Output tfrecord file path
    """
    print("(Dataset) Generating {}...".format(output_filepath))

    with tf.io.TFRecordWriter(output_filepath) as writer:
        inline, crossline, depth = seismic_data.shape
        print("(Dataset) \t|- There are {} samples...".format(inline))
        for i in range(inline):
            if i % 50 == 0:
                print("(Dataset) \t|- Processing {}th sample...".format(i))

            example = serialize_example(seismic_data[i, :, :], labels[i, :, :], crossline, depth)
            writer.write(example)

    print("(Dataset) Done generating {}.\n".format(output_filepath))


class Dataset:
    def __init__(self, config=Config.Config()):
        # Project configurations
        self.config = config

    def convert_numpy_to_tfrecord(self, overwrite=True):
        """
        Converts numpy datasets to tfrecord datasets for training, test #1, and test #2 sets.
        Note: dataset is of shape: (inline, crossline, depth)
        """
        # Convert testing sets
        self.convert_numpy_to_tfrecord_test_set(overwrite)

        # Convert training set
        self.convert_numpy_to_tfrecord_train_set(overwrite)

    def convert_numpy_to_tfrecord_train_set(self, overwrite=True):
        """
        Converts numpy dataset to tfrecord dataset for training set.
        """
        # Input training file paths (Numpy)
        train_data_fp = os.path.join(self.config.dataDirRaw, 'train', 'train_seismic.npy')
        train_labels_fp = os.path.join(self.config.dataDirRaw, 'train', 'train_labels.npy')

        # Output file directories (TFRecord)
        train_tfrecord_dir = os.path.join(self.config.dataDirConverted, 'train')
        valid_tfrecord_dir = os.path.join(self.config.dataDirConverted, 'valid')

        # Output file paths (TFRecord)
        train_tfrecord_fp = os.path.join(train_tfrecord_dir, 'train.tfrecord')
        valid_tfrecord_fp = os.path.join(valid_tfrecord_dir, 'valid.tfrecord')

        if not overwrite and os.path.exists(train_tfrecord_fp) and os.path.exists(train_tfrecord_fp):
            # TFRecords exists do not overwrite them
            return

        # Ensure directories exists
        if not os.path.isdir(train_tfrecord_dir):
            os.makedirs(train_tfrecord_dir)
        if not os.path.isdir(valid_tfrecord_dir):
            os.makedirs(valid_tfrecord_dir)

        # Load training data
        train_data = np.load(train_data_fp)
        train_labels = np.load(train_labels_fp)

        # Split data into training and validation data
        train_data_split, valid_data_split, train_labels_split, valid_labels_split = \
            train_test_split(train_data, train_labels, test_size=self.config.valRatio, random_state=self.config.seed)

        # Convert training set to TFRecord
        if not overwrite and os.path.exists(train_tfrecord_fp):
            write_dataset_to_tfrecord(train_data_split, train_labels_split, train_tfrecord_fp)

        # Convert validation set to TFRecord
        if not overwrite and os.path.exists(valid_tfrecord_fp):
            write_dataset_to_tfrecord(valid_data_split, valid_labels_split, valid_tfrecord_fp)

    def convert_numpy_to_tfrecord_test_set(self, overwrite=True):
        """
        Converts numpy dataset to tfrecord dataset for testing set.
        """

        # Output file directory (TFRecord)
        test_tfrecord_dir = os.path.join(self.config.dataDirConverted, 'test')

        # Ensure directories exists
        if not os.path.isdir(test_tfrecord_dir):
            os.makedirs(test_tfrecord_dir)

        # There are 2 testing sets.
        for i in range(1, 3):
            # Input file paths (Numpy)
            test_data_fp = os.path.join(self.config.dataDirRaw, 'test_once', 'test{}_seismic.npy'.format(i))
            test_labels_fp = os.path.join(self.config.dataDirRaw, 'test_once', 'test{}_labels.npy'.format(i))

            # Output file paths (TFRecord)
            test_tfrecord_fp = os.path.join(test_tfrecord_dir, 'test{}.tfrecord'.format(i))

            if not overwrite and os.path.exists(test_tfrecord_fp):
                # TFRecords exists do not overwrite them
                return

            # Load data
            test_data = np.load(test_data_fp)
            test_labels = np.load(test_labels_fp)

            # Convert to TFRecord
            write_dataset_to_tfrecord(test_data, test_labels, test_tfrecord_fp)
