

class Config:
    def __init__(self):
        # ----- Parameters -----
        # Path to original (numpy) dataset for Facies classification
        # From: https://github.com/yalaudah/facies_classification_benchmark
        self.data_dir_raw = './data'

        # Path to converted TFRecord dataset
        self.data_dir_converted = './data_converted'

        # How much of training data is used for validation
        self.val_ratio = 0.1

        # Fix random seed to have deterministic datasets
        self.seed = 2020

        # Class names
        self.class_names = ['Upper North Sea Group',
                            'Middle North Sea Group',
                            'Lower North Sea Group',
                            'Rijnland & Chalk Group',
                            'Scruff',
                            'Zechstein']

        # Training parameters
        self.num_epochs = 10
        self.shuffle_buffer_size = 2048
        self.batch_size = 4

        # ----- Constants -----
        self.num_classes = len(self.class_names)




