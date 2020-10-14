import Config
import Dataset
import MyModel
import tensorflow as tf
import Train


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

    # Convert numpy datasets to tfrecord datasets
    Dataset.convert_numpy_to_tfrecord(config, False)

    # Train model
    Train.train_model(config)




if __name__ == '__main__':
    main()
