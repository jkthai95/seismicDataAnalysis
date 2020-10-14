

class Config:
    def __init__(self):
        # Fix random seed to have deterministic datasets
        self.seed = 2020

        # Dataset parameters
        self.data_dir_raw = './data'    # Path to original (numpy) dataset for Facies classification
                                        # From: https://github.com/yalaudah/facies_classification_benchmark
        self.data_dir_converted = './data_converted'    # Path to converted TFRecord dataset
        self.class_names = ['Upper North Sea Group',    # Class names
                            'Middle North Sea Group',
                            'Lower North Sea Group',
                            'Rijnland & Chalk Group',
                            'Scruff',
                            'Zechstein']
        self.val_ratio = 0.1  # How much of training data is used for validation

        # Training parameters
        self.num_epochs = 100
        self.shuffle_buffer_size = 2048
        self.batch_size = 4
        self.drop_rate = 0.2
        self.patients = 20

        # Model parameters
        self.model_path = './model/model'      # Path to save and load existing model

        # Constants
        self.num_classes = len(self.class_names)




