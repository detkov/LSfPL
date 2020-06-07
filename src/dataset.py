from os.path import join
import numpy as np

import torch
from torch.utils.data import Dataset
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


transforms_train = A.Compose([
    # Rigid aug
    A.OneOf([
        A.ShiftScaleRotate(rotate_limit=1.0, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
    ], p=0.5),


    # Pixels aug
    A.OneOf([
        A.HueSaturationValue(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3)
    ], p=0.3),
    
    A.OneOf([
        A.IAAEmboss(p=1.0),
        A.IAASharpen(p=1.0),
        A.Blur(blur_limit=5, p=1.0),
        A.CLAHE(p=1.0)
    ], p=0.3),


    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

class MelanomaDataset(Dataset):
    def __init__(self, df, input_dir, transforms=None):
        self.df = df
        self.input_dir = input_dir
        self.transforms=transforms
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = join(self.input_dir, 'images_resized/', self.df.loc[idx, 'image_name'] + '.jpg')
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.loc[idx, 'target']
        label = label.astype(np.int8).reshape(-1,)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, label