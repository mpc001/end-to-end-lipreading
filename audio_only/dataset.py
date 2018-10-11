# encoding: utf-8
import time
import glob
import random
import scipy.io
import numpy as np


def load_file(filename):
    arrays = scipy.io.loadmat(filename)
    arrays = arrays['audio']
    arrays = arrays[0]
    arrays = arrays[0]
    arrays = arrays[0]
    arrays = arrays.flatten()
    arrays = arrays.astype(float)
    return arrays


class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        self.clean = 1 / 7.
        with open('../label_sorted.txt') as myfile:
            lines = myfile.read().splitlines()
        self.data_dir = [self.path + item for item in lines]
        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.mat')
        self.list = {}
        for i, x in enumerate(self.data_files):
            for j, elem in enumerate(self.data_dir):
                if elem in x:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs))/inputs_std

    def __getitem__(self, idx):
        noise_prop = (1-self.clean)/6.
        temp = random.random()
        if self.folds == 'train':
            if temp < noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/-5dB/'+self.list[idx][0][42:]
            elif temp < 2 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/0dB/'+self.list[idx][0][42:]
            elif temp < 3 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/5dB/'+self.list[idx][0][42:]
            elif temp < 4 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/10dB/'+self.list[idx][0][42:]
            elif temp < 5 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/15dB/'+self.list[idx][0][42:]
            elif temp < 6 * noise_prop:
                self.list[idx][0] = self.list[idx][0][:36]+'NoisyAudio/20dB/'+self.list[idx][0][42:]
            else:
                self.list[idx][0] = self.list[idx][0]
        elif self.folds == 'val' or self.folds == 'test':
                self.list[idx][0] = self.list[idx][0]
        inputs = load_file(self.list[idx][0])
        labels = self.list[idx][1]
        inputs = self.normalisation(inputs)
        return inputs, labels

    def __len__(self):
        return len(self.data_files)
