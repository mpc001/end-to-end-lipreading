# encoding: utf-8
import numpy as np
import glob
import time
import cv2


def load_file(filename):
    cap = np.load(filename)
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY)
                      for _ in xrange(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        with open('../label_sorted.txt') as myfile:
            lines = myfile.read().splitlines()
        self.data_dir = [self.path + item for item in lines]
        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.npy')
        self.list = {}
        for i, x in enumerate(self.data_files):
            for j, elem in enumerate(self.data_dir):
                if elem in x:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def __getitem__(self, idx):
        inputs = load_file(self.list[idx][0])
        labels = self.list[idx][1]
        return inputs, labels

    def __len__(self):
        return len(self.data_files)
