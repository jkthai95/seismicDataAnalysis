import Config
import Dataset
import numpy as np

def main():
    dataset = Dataset.Dataset()
    train_data, train_labels = dataset.loadRawTrainData()
    print('train_data.shape = {}'.format(train_data.shape))
    print('train_labels.shape = {}'.format(train_labels.shape))



if __name__ == '__main__':
    main()
