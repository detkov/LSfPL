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

from sklearn.metrics import roc_auc_score

from dataset import MelanomaDataset, train_transforms, test_transforms, tta_transforms

import warnings
warnings.filterwarnings('ignore')


def get_params(args):
    parser = argparse.ArgumentParser(description='Start fine-tuning process.')
    parser.add_argument('-c', '--config', help='Name of the config file.', 
                        dest='config')
    parser.add_argument('-l', '--last', help='Train only last layer of the model.', 
                        dest='train_only_last_layer')
    parser.add_argument('-f', '--file', help='Name of the train file with folds.', 
                        dest='folds_train_file')

    return parser.parse_args(args)


def main(params):
    folds_train_file = params.folds_train_file

    with open(join('../configs/', f'{params.config}.yaml'), 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print('Config params:')
    print(config)


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
    # SUBMISSIONS_DIR = config['SUBMISSIONS_DIR']
    SMOOTHEDLABELS_DIR = config['SMOOTHEDLABELS_DIR']
    exp_train_name = config['exp_train_name']
    hold_out_file = config['folds_train_file']
    os.makedirs(join(MODELS_DIR, folds_train_file), exist_ok=True)

    BS = config['batch_size']
    LR = config['learning_rate']
    EPOCHS = config['n_epochs']
    WORKERS = config['n_workers']
    WEIGHT_DECAY = config['weight_decay']

    N_FOLDS = config['n_folds']
    images_size = config['images_size']
    model_name = config['model_name'] # https://github.com/rwightman/pytorch-image-models
    device = config['device'] # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    gc.collect()


    df_train = pd.read_csv(join(SMOOTHEDLABELS_DIR, f'{folds_train_file}.csv'))
    train = MelanomaDataset(df_train, INPUT_DIR, images_size, train_transforms)

    # df_test = pd.read_csv(join(INPUT_DIR, 'sample_submission.csv'))
    # test = MelanomaDataset(df_test, INPUT_DIR, images_size, test_transforms)


    folds = list(range(N_FOLDS))
    for fold in folds:
        print(f'Fold: {fold}')

        model_path = join(MODELS_DIR, folds_train_file, f'fold_{fold}_weight.pth')
        model = timm.create_model(model_name, pretrained=True, num_classes=1)
        model = torch.load(join(MODELS_DIR, exp_train_name, f'fold_{fold}_weight.pth'))
        
        if params.train_only_last_layer:
            for parameter in model.parameters():
                parameter.requires_grad = False
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        model.cuda()
        
        optim = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, amsgrad=True)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(train, batch_size=BS, shuffle=True, num_workers=WORKERS)

        for epoch in range(EPOCHS):
            start_time = time.time()
            epoch_loss = 0
            
            model.train()
            for x, y in train_loader:
                x = torch.tensor(x, device=device, dtype=torch.float32)
                y = torch.tensor(y, device=device, dtype=torch.float32)
                optim.zero_grad()
                z = model(x)
                loss = criterion(z, y.unsqueeze(1))
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
            
            print(f'Epoch {epoch+1:02}: | Loss: {epoch_loss:.4f} | Training time: {str(datetime.timedelta(seconds=time.time() - start_time))[:7]}')
        
        torch.save(model, model_path)
        
        del train_loader, x, y
        torch.cuda.empty_cache()
        gc.collect()

    print('Getting result on hold-out set...')

    df = pd.read_csv(join(INPUT_DIR, f'{hold_out_file}.csv'))
    df_hold_out = df[df['kfold'] == -1].reset_index(drop=True)
    hold_out = MelanomaDataset(df_hold_out, INPUT_DIR, images_size, test_transforms)

    preds = torch.zeros((len(hold_out), 1), dtype=torch.float32, device=device)

    folds = list(range(N_FOLDS))
    for fold in folds:
        print(f'{fold} fold model:')
        hold_out_loader = DataLoader(hold_out, batch_size=BS, shuffle=False, num_workers=WORKERS)

        model_path = join(MODELS_DIR, folds_train_file, f'fold_{fold}_weight.pth')
        model = torch.load(model_path)
        model.eval()
        
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
        with torch.no_grad():
            for i, (x_test, _) in enumerate(hold_out_loader):
                x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
                z_test = tta_model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i*x_test.shape[0]:i*x_test.shape[0] + x_test.shape[0]] += z_test
    preds /= len(folds)
    preds = preds.cpu().detach().numpy()

    hold_out_auc = roc_auc_score(df_hold_out['target'].values, preds)
    os.mknod(join(MODELS_DIR, folds_train_file, f'roc-auc:{hold_out_auc:.4f}'))
    print(f'\nROC AUC on hold-out set: {hold_out_auc:.4f}')
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main(get_params(sys.argv[1:]))