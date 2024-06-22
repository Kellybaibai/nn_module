# -*- coding: utf-8 -*-
# @Time: 2024/6/22 14:51
# @Author: Kellybai
# @File: data_process.py
# Have a nice day!

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

np.random.seed(2024)

class MyDataset:
    def __int__(self, args):
        self.data_name = args.dataset
    def data_load(self):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.data_name == 'iris':
            dataset = load_iris(root='./data')
            train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=2024)
        elif self.data_name == 'MNIST':
            train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
            test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        else:
            train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        return train_dataset, test_dataset


class MyDataLoader(DataLoader):
    def __int__(self, args, dataset, shuffle=True):
        self.dataset = dataset
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.len_data = len(dataset)
        self.current_idx = 0

        if self.shuffle == True:
            np.random.shuffle(self.dataset)

    def __iter__(self):
        '''
        to make MyDataloader iterable
        '''
        return self

    def __next__(self):
        if self.current_idx >= self.len_data:
            raise StopIteration

        batch_data = self.dataset[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        batch_data = torch.Tensor(batch_data)
        return batch_data

    def __len__(self):
        '''
        to calculate the number of batches
        '''
        return self.len_data // self.batch_size + (1 if self.len_data % self.batch_size > 0 else 0)








