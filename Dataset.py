import Config
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob


def crop_data_for_even_shape(seismic_data, labels):
    """
    Make shape of each image even for neural network.
    :param seismic_data: numpy array of shape (inline, crossline, depth)
    :param labels: numpy array of shape (inline, crossline, depth)
    :return: cropped seismic_data and labels
    """
    inline, crossline, depth = seismic_data.shape
    new_crossline = 2 * (crossline // 2)
    new_depth = 2 * (depth // 2)

    return seismic_data[:, :new_crossline, :new_depth], labels[:, :new_crossline, :new_depth]


def serialize_example(seismic, label, width, height):
    """
    Creates a tf.train.Example message ready to be written to a file.
    :param seismic: Seismic image.
    :param label:   Label image.
    :param width:   Width of image.
    :param height:  Height of image.
    :return:        tf.train.Example message.
    """
    # Create dictionary mapping for each feature
    feature = {
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
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

    # Make image shape even for neural network
    seismic_data, labels = crop_data_for_even_shape(seismic_data, labels)

    with tf.io.TFRecordWriter(output_filepath) as writer:
        inline, crossline, depth = seismic_data.shape
        print("(Dataset) \t|- There are {} samples...".format(inline))
        for i in range(inline):
            if i % 50 == 0:
                print("(Dataset) \t|- Processing {}th sample...".format(i))

            example = serialize_example(seismic_data[i, :, :], labels[i, :, :], crossline, depth)
            writer.write(example)

    print("(Dataset) Done generating {}.\n".format(output_filepath))


# Feature description used to parse TFRecords.
feature_description = {
    'width': tf.io.FixedLenFeature([], tf.int64),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'seismic': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'label': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
}


def parse_tfrecord(example_proto):
    """
    Parses serialized example into a dictionary of tensors.
    :param example_proto: Serialized example.
    :return: Dictionary of tensors.
    """
    example = tf.io.parse_example(example_proto, feature_description)
    example['seismic'] = tf.reshape(example['seismic'], [example['width'], example['height'], 1])
    example['labels'] = tf.reshape(example['seismic'], [example['width'], example['height'], 1])

    return example


def normalize_labels(labels, num_classes):
    """
    Normalizes labels between [-1, 1] based on number of classes.
    :param labels: Default labels.
    :param num_classes: Number of classes.
    :return: Normalized labels.
    """
    normalized_labels = labels.astype(np.float)
    normalized_labels = 2 * normalized_labels / (num_classes - 1) - 1
    return normalized_labels

class Dataset:
    def __init__(self, config=Config.Config()):
        # Project configurations
        self.config = config

    def acquire_tfrecord_dataset(self, partition):
        """
        Acquire tf.data.TFRecordDataset for specific partition.
        :param partition: Which partition? "train", "valid", "test1", or "test2"
        :return: tf.data.TFRecordDataset for specific partition.
        """
        is_train_data = False

        if 'train' in partition:
            is_train_data = True
            filenames = glob.glob(os.path.join(self.config.data_dir_converted, 'train', "*.tfrecord"))
        elif 'valid' in partition:
            filenames = glob.glob(os.path.join(self.config.data_dir_converted, 'valid', "*.tfrecord"))
        elif 'test1' in partition:
            filenames = glob.glob(os.path.join(self.config.data_dir_converted, 'test1', "*.tfrecord"))
        elif 'test2' in partition:
            filenames = glob.glob(os.path.join(self.config.data_dir_converted, 'test2', "*.tfrecord"))

        # Create TFRecord Dataset
        tfrecord_dataset = tf.data.TFRecordDataset(filenames)
        tfrecord_dataset = tfrecord_dataset.map(parse_tfrecord)

        if is_train_data:
            # Cache parsed data into memory
            tfrecord_dataset = tfrecord_dataset.cache()

            # shuffle dataset
            tfrecord_dataset = tfrecord_dataset.shuffle(self.config.shuffle_buffer_size)

        # Batch data
        tfrecord_dataset = tfrecord_dataset.batch(self.config.batch_size)

        if not is_train_data:
            # Cache dataset into memory
            tfrecord_dataset = tfrecord_dataset.cache()

        return tfrecord_dataset

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
        train_data_fp = os.path.join(self.config.data_dir_raw, 'train', 'train_seismic.npy')
        train_labels_fp = os.path.join(self.config.data_dir_raw, 'train', 'train_labels.npy')

        # Output file directories (TFRecord)
        train_tfrecord_dir = os.path.join(self.config.data_dir_converted, 'train')
        valid_tfrecord_dir = os.path.join(self.config.data_dir_converted, 'valid')

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

        # Normalize labels
        train_labels = normalize_labels(train_labels, self.config.num_classes)

        # Split data into training and validation data
        train_data_split, valid_data_split, train_labels_split, valid_labels_split = \
            train_test_split(train_data, train_labels, test_size=self.config.val_ratio, random_state=self.config.seed)

        # Convert training set to TFRecord
        if overwrite or not os.path.exists(train_tfrecord_fp):
            write_dataset_to_tfrecord(train_data_split, train_labels_split, train_tfrecord_fp)

        # Convert validation set to TFRecord
        if overwrite or not os.path.exists(valid_tfrecord_fp):
            write_dataset_to_tfrecord(valid_data_split, valid_labels_split, valid_tfrecord_fp)

    def convert_numpy_to_tfrecord_test_set(self, overwrite=True):
        """
        Converts numpy dataset to tfrecord dataset for testing set.
        """
        # There are 2 testing sets.
        for i in range(1, 3):
            # Output file directory (TFRecord)
            test_tfrecord_dir = os.path.join(self.config.data_dir_converted, 'test{}'.format(i))

            # Ensure directories exists
            if not os.path.isdir(test_tfrecord_dir):
                os.makedirs(test_tfrecord_dir)

            # Input file paths (Numpy)
            test_data_fp = os.path.join(self.config.data_dir_raw, 'test_once', 'test{}_seismic.npy'.format(i))
            test_labels_fp = os.path.join(self.config.data_dir_raw, 'test_once', 'test{}_labels.npy'.format(i))

            # Output file paths (TFRecord)
            test_tfrecord_fp = os.path.join(test_tfrecord_dir, 'test{}.tfrecord'.format(i))

            if not overwrite and os.path.exists(test_tfrecord_fp):
                # TFRecords exists do not overwrite them
                return

            # Load data
            test_data = np.load(test_data_fp)
            test_labels = np.load(test_labels_fp)

            # Normalize labels
            test_labels = normalize_labels(test_labels, self.config.num_classes)

            # Convert to TFRecord
            write_dataset_to_tfrecord(test_data, test_labels, test_tfrecord_fp)
