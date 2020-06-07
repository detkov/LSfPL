from os.path import join
import numpy as np 
import pandas as pd 
import gc
from tqdm import tqdm
import cv2

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dataset import MelanomaDataset, train_transforms, test_transforms

import os
import random


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    ######
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    os.makedirs('models/', exist_ok=True)
    torch.cuda.empty_cache()
    gc.collect()

    ######
    seed_everything(42)
    INPUT_DIR = '../input/'

    N_CLASSES = 2
    BS = 10
    LR = 3e-4
    EPOCHS = 10
    WORKERS = 8
    model_name = 'tf_efficientnet_b3_ns' # https://github.com/rwightman/pytorch-image-models

    df_train = pd.read_csv(join(INPUT_DIR, 'train_folds.csv'))
    df_sample = pd.read_csv(join(INPUT_DIR, 'sample_submission.csv'))

    ######
    def train_model(model, epoch):
        model.train() 
        
        losses = AverageMeter()
        avg_loss = 0.

        optimizer.zero_grad()
        
        tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_train, labels_train = imgs.cuda(), labels.cuda().long()
            output_train = model(imgs_train)

            loss = criterion(output_train, labels_train)
            loss.backward()

            optimizer.step() 
            optimizer.zero_grad() 
            
            avg_loss += loss.item() / len(train_loader)
            
            losses.update(loss.item(), imgs_train.size(0))

            tk.set_postfix(loss=losses.avg)
            
        return avg_loss


    def test_model(model):    
        model.eval()
        
        losses = AverageMeter()
        avg_val_loss = 0.
        
        valid_preds, valid_targets = [], []
        
        with torch.no_grad():
            tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            for idx, (imgs, labels) in enumerate(tk):
                imgs_valid, labels_valid = imgs.cuda(), labels.cuda().long()
                output_valid = model(imgs_valid)
                
                loss = criterion(output_valid, labels_valid)
                
                avg_val_loss += loss.item() / len(val_loader)

                losses.update(loss.item(), imgs_valid.size(0))
                
                tk.set_postfix(loss=losses.avg)
                
                valid_preds.append(torch.softmax(output_valid,1)[:,1].detach().cpu().numpy())
                valid_targets.append(labels_valid.detach().cpu().numpy())
                
            valid_preds = np.concatenate(valid_preds)
            valid_targets = np.concatenate(valid_targets)
            auc =  roc_auc_score(valid_targets, valid_preds) 
                
        return avg_val_loss, auc

    ######
    folds = [0,1,2,3,4]
    cv = []

    for fold in folds:
        print(f'Fold: {fold}')
        folds_to_train = list(set(folds)-set([fold]))
        
        train_df = df_train[df_train['kfold'].isin(folds_to_train)].reset_index(drop=True)
        valid_df = df_train[df_train['kfold'] == fold].reset_index(drop=True)

        trainset = MelanomaDataset(train_df, INPUT_DIR, transforms=train_transforms)
        train_loader = DataLoader(trainset, batch_size=BS, shuffle=True, num_workers=WORKERS)
    
        valset = MelanomaDataset(valid_df, INPUT_DIR, test_transforms)
        val_loader = DataLoader(valset, batch_size=BS, shuffle=False, num_workers=WORKERS)

        model = timm.create_model(model_name, pretrained=True, num_classes=N_CLASSES)
        model.cuda()

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=2, gamma=0.3)

        best_auc = 0
        es = 0

        for epoch in range(EPOCHS):
            avg_loss = train_model(model, epoch)
            avg_val_loss, auc = test_model(model)

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f'models/fold_{fold}_weight.pth')
            else:
                es += 1
                if es > 1:
                    break
            print('Current Valid AUC:', auc, 'Best Valid AUC:', best_auc)
            scheduler.step()

        cv.append(best_auc)
    print('CV AUC scores:', ' | '.join(map(lambda x: str(round(x, 4)), cv)))
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()