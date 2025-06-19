import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as TTF
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import torch
# import matplotlib.pyplot as plt


class Car_Dataset(Dataset):
    def __init__(self, data_paths, data_labels, data_bboxes, transform=None, Crop = True):
        super().__init__()

        self.data_paths = data_paths
        self.data_labels = data_labels
        self.data_bboxes = data_bboxes #bboxes按照x1,y1,x2,y2的顺序输入
        self.transform = transform
        self.Crop = Crop

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data_label = self.data_labels[index]
        bbox_x1 = self.data_bboxes[index][0]
        bbox_y1 = self.data_bboxes[index][1]
        bbox_x2 = self.data_bboxes[index][2]
        bbox_y2 = self.data_bboxes[index][3]


        images = Image.open(data_path).convert('RGB')
        if self.Crop:
            images = TTF.crop(images, bbox_y1, bbox_x1, (bbox_y2 - bbox_y1), (bbox_x2 - bbox_x1))

        if self.transform:
            images = self.transform(images)

        
        labels = torch.tensor(data_label).long()

        return images, labels