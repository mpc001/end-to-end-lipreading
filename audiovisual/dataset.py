#encoding: utf-8
import os
import cv2
import glob
import random
import scipy.io
import numpy as np


def load_audio_file(filename):
    arrays = scipy.io.loadmat(filename)
    arrays = arrays['audio']
    arrays = arrays[0]
    arrays = arrays[0]
    arrays = arrays[0]
    arrays = arrays.flatten()
    arrays = arrays.astype(float)
    return arrays


def load_video_file(filename):
    cap = np.load(filename)
    arrays = np.stack([cv2.cvtColor(cap[_], cv2.COLOR_RGB2GRAY) for _ in xrange(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class MyDataset():
    def __init__(self, folds, audio_path, video_path):
        self.folds = folds
        self.audio_path = audio_path
        self.video_path = video_path
        self.clean = 1 / 7.
        with open('../label_sorted.txt') as myfile:
            lines = myfile.read().splitlines()
        self.data_dir = [self.audio_path + item for item in lines]
        self.data_files = glob.glob(self.audio_path+'*/'+self.folds+'/*.mat')
        self.list = {}
        for i, x in enumerate(self.data_files):
            for j, elem in enumerate(self.data_dir):
                if elem in x:
                    self.list[i] = [x]
                    self.list[i].append(j)

    def normalisation(self, inputs):
        inputs_std = np.std(inputs)
        if inputs_std == 0.:
            inputs_std = 1.
        return (inputs - np.mean(inputs))/inputs_std

    def __getitem__(self, idx):
        video_inputs = load_video_file(self.video_path+self.list[idx][0][42:-4]+'.npy')
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
        audio_inputs = load_audio_file(self.list[idx][0])
        audio_inputs = self.normalisation(audio_inputs)
        labels = self.list[idx][1]
        return audio_inputs, video_inputs, labels

    def __len__(self):
        return len(self.data_files)
