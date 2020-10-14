import Config
import Dataset
import MyModel
import tensorflow as tf
import numpy as np
import os


def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def main():
    # Limit GPU usage
    limit_gpu()

    # Project configurations
    config = Config.Config()

    # Initialize class to handle dataset
    dataset = Dataset.Dataset(config)

    # Convert numpy datasets to tfrecord datasets
    dataset.convert_numpy_to_tfrecord(False)

    # Acquire TFRecord dataset
    tfrecord_dataset = dataset.acquire_tfrecord_dataset('train')

    # Acquire input shape
    for example in tfrecord_dataset.take(1):
        input_shape = example['seismic'].shape

    # Define model
    model = MyModel.MyModel(drop_rate=0.2)
    inputs = tf.keras.Input(shape=input_shape)
    model = model.get_model(inputs)

    # Verify shapes are the same
    for example in tfrecord_dataset.take(1):
        print("input sample shape = {}".format(example['seismic'].shape))
        model_input_shape = list(input_shape)
        model_input_shape.insert(0, 1)
        # print("model output shape = {}".format(model(example['seismic']).shape))
        print("model output shape = {}".format(model(tf.reshape(example['seismic'], model_input_shape)).shape))


if __name__ == '__main__':
    main()
