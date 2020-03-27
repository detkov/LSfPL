import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


DIR_INPUT = '../input'

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


    # Non-rigid aug
    A.OneOf([
        A.ElasticTransform(p=1.0),
        A.IAAPiecewiseAffine(p=1.0),
        A.GridDistortion(distort_limit=0.6, p=1.0),
        A.OpticalDistortion(distort_limit=0.7, shift_limit=0.2, p=1.0)
    ], p=0.5),

    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

transforms_valid = A.Compose([
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
])

class PlantDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms=transforms

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = DIR_INPUT + '/images_res/' + self.df.loc[idx, 'image_id'] + '.jpg'
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = self.df.loc[idx, ['healthy', 'multiple_diseases', 'rust', 'scab']].values
        labels = torch.from_numpy(labels.astype(np.int8))
        labels = labels.unsqueeze(-1)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']

        return image, labels