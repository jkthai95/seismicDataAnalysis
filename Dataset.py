import Config
import os
import numpy as np

class Dataset:
    def __init__(self, config=Config.Config()):
        # Project configurations
        self.config = config

    def loadRawTrainData(self):
        # Train data filepath
        train_data_fp = os.path.join(self.config.dataDirRaw, 'train', 'train_seismic.npy')
        train_labels_fp = os.path.join(self.config.dataDirRaw, 'train', 'train_labels.npy')

        # Load train data
        train_data = np.load(train_data_fp)
        train_labels = np.load(train_labels_fp)

        return train_data, train_labels
