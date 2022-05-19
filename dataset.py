import os
import torch
import sys
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = os.listdir(path)
        self.file_number = len(self.files)
        self.path = path
        self.in_len = 10
        self.out_len = 20
        self.solution = 896

    def __len__(self):
        return self.file_number

    def __getitem__(self, idx):
        image_file_path = os.path.join(self.path, self.files[idx])
        images = os.listdir(image_file_path)
        images = sorted(images, key=lambda x: int(x[:12]))
        image_length = len(images)
        inputs = np.ones([image_length, self.solution, self.solution, 1])
        for i in range(image_length):
            img_path = os.path.join(image_file_path, images[i])
            img = Image.open(img_path)
            img = np.array(img)
            img = cv2.resize(img, (self.solution, self.solution),
                             interpolation=cv2.INTER_NEAREST).reshape(self.solution, self.solution, 1)
            mask = (img > 80)
            img = (1 - mask) * img + 0 * mask
            img = img / 80.0
            inputs[i] = img
        input_seq = inputs[0:self.in_len + self.out_len, :, :, :]

        return input_seq


class TestDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.files = os.listdir(path)
        self.file_number = len(self.files)
        self.path = path
        self.in_len = 10
        self.out_len = 20
        self.solution = 896

    def __len__(self):
        return self.file_number

    def __getitem__(self, idx):
        image_file_path = os.path.join(self.path, self.files[idx])
        images = os.listdir(image_file_path)
        images = sorted(images, key=lambda x: int(x[:12]))
        image_length = len(images)
        inputs = np.ones([image_length, self.solution, self.solution, 1])
        for i in range(image_length):
            img_path = os.path.join(image_file_path, images[i])
            img = Image.open(img_path)
            img = np.array(img)
            img = cv2.resize(img, (self.solution, self.solution),
                             interpolation=cv2.INTER_NEAREST).reshape(self.solution, self.solution, 1)
            mask = (img > 80)
            img = (1 - mask) * img + 0 * mask
            img = img / 80.0
            inputs[i] = img
        input_seq = inputs[0:self.in_len + self.out_len, :, :, :]
        return input_seq