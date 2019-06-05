# encoding: utf-8
import os
import cv2
import glob
import numpy as np


def load_file(filename):
    cap = np.load(filename)['data']
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY)
                      for _ in xrange(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        with open('../label_sorted.txt') as myfile:
            self.data_dir = myfile.read().splitlines()
        self.filenames = glob.glob(os.path.join(self.path, '*', self.folds, '*.npz'))
        self.list = {}
        for i, x in enumerate(self.filenames):
            target = x.split('/')[-3]
            for j, elem in enumerate(self.data_dir):
                if elem == target:
                    self.list[i] = [x]
                    self.list[i].append(j)
        print('Load {} part'.format(self.folds))

    def __getitem__(self, idx):
        inputs = load_file(self.list[idx][0])
        labels = self.list[idx][1]
        return inputs, labels

    def __len__(self):
        return len(self.filenames)
