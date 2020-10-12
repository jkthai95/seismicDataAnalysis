import Config
import Dataset
import tensorflow as tf

def main():
    # Project configurations
    config = Config.Config()

    # Initialize class to handle dataset
    dataset = Dataset.Dataset(config)

    # Convert numpy datasets to tfrecord datasets
    dataset.convert_numpy_to_tfrecord(False)


if __name__ == '__main__':
    main()
