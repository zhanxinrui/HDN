import torch
import numpy as np
import cv2

class Normalize(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, sample):
        return (sample / 255. - self.mean) / self.std

class ToTensor(object):
    def __call__(self, sample):
        sample = sample.transpose(2, 0, 1)
        return torch.from_numpy(sample.astype(np.float32))
