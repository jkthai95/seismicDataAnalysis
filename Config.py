

class Config:
    def __init__(self):
        # Path to original (numpy) dataset for Facies classification
        # From: https://github.com/yalaudah/facies_classification_benchmark
        self.dataDirRaw = './data'

        # Path to converted TFRecord dataset
        self.dataDirConverted = './data_converted'

        # How much of training data is used for validation
        self.valRatio = 0.1

        # Fix random seed to have deterministic datasets
        self.seed = 2020
