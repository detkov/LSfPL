from os.path import join
import numpy as np

from torch.utils.data import Dataset
import cv2

import albumentations as A
from albumentations.pytorch import ToTensor
import ttach as tta


train_transforms = A.Compose([
    # Rigid aug
    A.OneOf([
        A.ShiftScaleRotate(rotate_limit=90, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ], p=0.5),

    # Pixels aug
    A.OneOf([
        A.RandomBrightness(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.RandomGamma(p=1.0),
    ], p=0.5),



    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensor()
])

test_transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensor()
])

tta_transforms = tta.aliases.d4_transform()


class MelanomaDataset(Dataset):
    def __init__(self, df, input_dir, images_size, transforms):
        self.df = df
        self.input_dir = input_dir
        self.images_size = images_size
        self.transforms = transforms
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        image_src = join(self.input_dir, f'images_resized_{self.images_size}/', self.df.loc[idx, 'image_name'] + '.jpg')            
        image = cv2.imread(image_src)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = self.df.loc[idx, 'target']
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return image, label