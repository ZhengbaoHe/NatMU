### 数据集获取
import os

import torchvision
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import time, random


def unpickle(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


transform_train_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()  # 会将数据归一化0-1
])
transform_valid_cifar = transforms.Compose([
    transforms.ToTensor()  # 会将数据归一化0-1
])


CIFAR100_CLASS_DICT = {
    "rocket": 69,
    "vehicle2": 19, # superclass
    "cattle": 19,
    "veg": 4,
    "mushroom": 51,
    "people": 14,
    "baby": 2,
    "electrical_devices": 5,
    "lamp": 40,
    "natural_scenes": 10,
    "sea": 71,
    "42": 42,
    "1": 1,
    "10": 10,
    "20": 20,
    "30": 30,
    "40": 40,
}

COARSE_MAP = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

CLASS_MAP_ARRAY = [0] * 100
for i in range(100):
    for key in range(20):
        if i in COARSE_MAP[key]: CLASS_MAP_ARRAY[i] = key
CLASS_MAP_ARRAY = np.array(CLASS_MAP_ARRAY, dtype=np.long)

class DatasetFullClassUnlearning(Dataset):
    def __init__(self, transform, forget_class=-1, isForgetSet=False):
        self.forget_class = forget_class
        self.isForgetSet = isForgetSet
        self.transform = transform
        
    def __getitem__(self, index):
        tmp_data = self.data[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform(img), self.label[index]

    def __len__(self):
        return self.data.shape[0]
    
    def process_data(self):
        forget_idx = np.nonzero(self.label == self.forget_class)[0]
        if not self.isForgetSet:
            self.data = np.delete(self.data, forget_idx, axis=0).copy()
            self.label = np.delete(self.label, forget_idx, axis=0).copy()
        else:
            self.data = self.data[forget_idx].copy()
            self.label = self.label[forget_idx].copy()
        
class DatasetCifar100FullClassUnlearning(DatasetFullClassUnlearning):
    def __init__(self, mode, transform, forget_class=-1, isForgetSet=False, ):
        if mode == "train":
            dict = unpickle('../data/cifar100/train')
        elif mode == "test":
            dict = unpickle('../data/cifar100/test')
        else:
            raise KeyError("Cifar100 Dataset mode error")    
        self.data = dict[b'data'].reshape(-1, 3, 32, 32)
        self.label = np.array(dict[b'fine_labels'])    
        super().__init__(transform=transform, forget_class=forget_class, isForgetSet=isForgetSet)
        self.process_data()
    
class DatasetCifar20FullClassUnlearning(DatasetCifar100FullClassUnlearning):
    def __init__(self, mode, transform, forget_class=-1, isForgetSet=False, ):
        super().__init__(mode=mode, transform=transform, forget_class=forget_class, isForgetSet=isForgetSet)
        self.label = CLASS_MAP_ARRAY[self.label]


class ConcatDataset(Dataset):
    def __init__(self, data1, label1, data2, label2, transform1, transform2):
        self.data1  = data1.copy()
        self.label1 = label1.copy()
        self.data2  = data2.copy()
        self.label2 = label2.copy()
        
        self.transform1 = transform1
        self.transform2 = transform2
        
    def __getitem__(self, index):
        if index < self.data1.shape[0]:   
            tmp_data = self.data1[index]
            img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
            return self.transform1(img), self.label1[index], True
        
        index = index - self.data1.shape[0]
        tmp_data = self.data2[index]
        img = Image.fromarray(np.transpose(tmp_data, (1, 2, 0)))
        return self.transform2(img), self.label2[index], False

    def __len__(self):
        return self.data1.shape[0] + self.data2.shape[0]
    


