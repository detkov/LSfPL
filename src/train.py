from os.path import join
import numpy as np 
import pandas as pd 
import gc
from tqdm import tqdm
import cv2
import time
import datetime
import yaml
import os
import random
import sys
import argparse

import timm
import ttach as tta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from sklearn.metrics import roc_auc_score

from dataset import MelanomaDataset, train_transforms, test_transforms

import warnings
warnings.filterwarnings('ignore')


def get_params(args):
    parser = argparse.ArgumentParser(description='Start training process')
    parser.add_argument('-c', '--config', help='Name of the config file.', 
                        dest='config')
    return parser.parse_args(args)


def main(params):
    ###
    exp_name = params.config
    with open(join('../configs/', f'{exp_name}.yaml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print('Config params:')
    print(config)


    ###
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    seed_everything(config['seed'])
    INPUT_DIR = config['INPUT_DIR']
    MODELS_DIR = config['MODELS_DIR']
    SUBMISSIONS_DIR = config['SUBMISSIONS_DIR']
    os.makedirs(join(MODELS_DIR, exp_name), exist_ok=True)
    try: os.mknod(join(MODELS_DIR, exp_name, f'{exp_name}.txt'))
    except: pass

    BS = config['batch_size']
    LR = config['learning_rate']
    EPOCHS = config['n_epochs']
    WORKERS = config['n_workers']
    ES_PATIENCE = config['early_stopping_patience']
    REDUCELR_PATIENCE = config['reduce_lr_on_plateau_patience']
    REDUCELR_FACTOR = config['reduce_lr_on_plateau_factor']
    # STEP_SIZE = config['steplr_step_size']

    N_FOLDS = config['n_folds']
    images_size = config['images_size']
    model_name = config['model_name'] # https://github.com/rwightman/pytorch-image-models
    device = config['device'] # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    gc.collect()


    ###
    df_train = pd.read_csv(join(INPUT_DIR, f'{config["folds_train_file"]}.csv'))
    df_test = pd.read_csv(join(INPUT_DIR, 'sample_submission.csv'))
    test = MelanomaDataset(df_test, INPUT_DIR, images_size, test_transforms)


    ###
    preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)

    folds = list(range(N_FOLDS))
    for fold in folds:
        print(f'Fold: {fold}')
        folds_to_train = list(set(folds)-set([fold]))
        
        best_val = None 
        patience = ES_PATIENCE
        model = timm.create_model(model_name, pretrained=True, num_classes=1)
        model.cuda()
        
        optim = AdamW(model.parameters(), lr=LR, amsgrad=True)
        # scheduler = StepLR(optim, step_size=STEP_SIZE, gamma=0.3)
        scheduler = ReduceLROnPlateau(optim, mode='max', patience=REDUCELR_PATIENCE, verbose=True, factor=REDUCELR_FACTOR)
        criterion = nn.BCEWithLogitsLoss()

        train_df = df_train[df_train['kfold'].isin(folds_to_train)].reset_index(drop=True)
        valid_df = df_train[df_train['kfold'] == fold].reset_index(drop=True)

        train = MelanomaDataset(train_df, INPUT_DIR, images_size, train_transforms)
        val = MelanomaDataset(valid_df, INPUT_DIR, images_size, test_transforms)
        
        train_loader = DataLoader(train, batch_size=BS, shuffle=True, num_workers=WORKERS)
        val_loader = DataLoader(val, batch_size=BS, shuffle=False, num_workers=WORKERS)

        model_path = join(MODELS_DIR, exp_name, f'fold_{fold}_weight.pth')

        for epoch in range(EPOCHS):
            start_time = time.time()
            epoch_loss = 0
            
            model.train()
            for i, (x, y) in enumerate(tqdm(train_loader, total=len(train_loader), position=0, leave=True)):
                x = torch.tensor(x, device=device, dtype=torch.float32)
                y = torch.tensor(y, device=device, dtype=torch.float32)
                optim.zero_grad()
                z = model(x)
                loss = criterion(z, y.unsqueeze(1))
                loss.backward()
                optim.step()
                epoch_loss += loss.item()

            model.eval()
            val_preds = torch.zeros((len(valid_df), 1), dtype=torch.float32, device=device)
            with torch.no_grad():
                for j, (x_val, y_val) in enumerate(tqdm(val_loader, total=len(val_loader), position=0, leave=True)):
                    x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
                    y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
                    z_val = model(x_val)
                    val_pred = torch.sigmoid(z_val)
                    val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0]] = val_pred
                val_auc = roc_auc_score(valid_df['target'].values, val_preds.cpu().detach())
                
                print('Epoch {:02}: | Loss: {:.4f} | Val roc_auc: {:.4f} | Training time: {}'.format(
                epoch+1, 
                epoch_loss, 
                val_auc, 
                str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
                
                scheduler.step(val_auc)
                # During the first iteration (first epoch) best validation is set to None
                if not best_val:
                    best_val = val_auc
                    torch.save(model, model_path) 
                    continue
                    
                if val_auc >= best_val:
                    best_val = val_auc
                    patience = ES_PATIENCE
                    torch.save(model, model_path)
                else:
                    patience -= 1
                    if patience == 0:
                        print('Early stopping. Best Val roc_auc: {:.4f}'.format(best_val))
                        break
        
        test_loader = DataLoader(test, batch_size=BS, shuffle=False, num_workers=WORKERS)
        model = torch.load(model_path)
        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.d4_transform())
        with torch.no_grad():        
            for i, (x_test, _) in enumerate(tqdm(test_loader, total=len(test_loader), position=0, leave=True)):
                x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
                z_test = tta_model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i*x_test.shape[0]:i*x_test.shape[0] + x_test.shape[0]] += z_test

        del train, val, train_loader, val_loader, train_df, valid_df, x, y, x_val, y_val, x_test, _
        torch.cuda.empty_cache()
        gc.collect()

    preds /= len(folds)


    ###
    print('Making submission...')
    sub = pd.read_csv(join(INPUT_DIR, 'sample_submission.csv'))
    sub['target'] = preds.cpu().detach().numpy().reshape(-1,)
    sub.to_csv(join(SUBMISSIONS_DIR, f'{exp_name}.csv'), index=False)
    torch.cuda.empty_cache()
    gc.collect()
    print('Submission is created...')

    ###
    print('Getting result on hold-outed set...')

    df = pd.read_csv(join(INPUT_DIR, f'{config["folds_train_file"]}.csv'))
    df_hold_out = df[df['kfold'] == -1].reset_index(drop=True)
    hold_out = MelanomaDataset(df_hold_out, INPUT_DIR, images_size, test_transforms)

    preds = torch.zeros((len(hold_out), 1), dtype=torch.float32, device=device)

    for fold in folds:
        print(f'{fold} fold model:')
        hold_out_loader = DataLoader(hold_out, batch_size=BS, shuffle=False, num_workers=WORKERS)

        model_path = join(MODELS_DIR, exp_name, f'fold_{fold}_weight.pth')
        model = torch.load(model_path)
        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
        with torch.no_grad():
            for i, (x_test, _) in enumerate(tqdm(hold_out_loader, total=len(hold_out_loader), position=0, leave=True)):
                x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
                z_test = tta_model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i*x_test.shape[0]:i*x_test.shape[0] + x_test.shape[0]] += z_test
    preds /= len(folds)
    preds = preds.cpu().detach().numpy()

    hold_out_auc = roc_auc_score(df_hold_out['target'].values, preds)
    print(f'\nROC AUC on hold-outed set: {hold_out_auc:.4f}')


if __name__ == "__main__":
    main(get_params(sys.argv[1:]))