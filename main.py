import argparse
import numpy as np
from src.data_process import MyDataset, MyDataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='iris', help='[iris,MNIST,Fashion]')
    parser.add_argument('--batch_size',type=int, default=128)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args =get_args()

    # Load the Data
    Dataset = MyDataset(args)
    train_dataset, test_dataset = Dataset.data_load()
    train_dataloader = MyDataLoader(args, train_dataset)
    test_dataloader = MyDataLoader(args, test_dataset)

    #





