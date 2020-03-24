import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


DIR_INPUT = '../input'

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