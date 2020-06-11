import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold


def get_params(args):
    parser = argparse.ArgumentParser(description='Makes a split of your train file with non-overlapping groups.')
    parser.add_argument('-i', '--input', help='Path to `train.csv`.', 
                        dest='input', type=str, default='../input/train.csv')
    parser.add_argument('-o','--output', help='Path to `train.csv` with folds, passing arg adds `_folds_groups` to the name of the input file.', 
                        dest='output', type=str)
    parser.add_argument('-s', '--splits', help='Number of splits.', 
                        dest='splits', type=int, default=5)
    parser.add_argument('-v', '--valid', help='Percent of data to be hold-outed for validation (Ensembling, WAE, etc.).', 
                        dest='valid', type=int, default=0)
    parser.add_argument('-f', '--feature', help='Unique feature of row, f.e. "image_name".', 
                        dest='feature', type=str, required=True)
    parser.add_argument('-t', '--target', help='Target feature to be splitted by.', 
                        dest='target', type=str, default='target')
    parser.add_argument('-g', '--group', help='Group feature to be splitted by.', 
                        dest='group', type=str, default='group')
    return parser.parse_args(args)


def main(params):
    df = pd.read_csv(params.input)
    df['kfold'] = np.nan

    X = df[params.feature].values
    y = df[params.target].values
    groups = df[params.group].values

    if params.valid != 0:
        gkf = GroupKFold(n_splits=int(100/params.valid))
        for fold, (_, valid) in enumerate(gkf.split(X, y, groups)):
            if fold == 0:
                df.loc[valid, 'kfold'] = -1
            else:
                break
    df_valid = df[df['kfold'] == -1].reset_index(drop=True)


    df_train = df[df['kfold'].isnull()].reset_index(drop=True)
    X = df_train[params.feature].values
    y = df_train[params.target].values
    groups = df_train[params.group].values

    gkf = GroupKFold(n_splits=params.splits)
    for fold, (_, valid) in enumerate(gkf.split(X, y, groups)):
        df_train.loc[valid, 'kfold'] = fold


    df = pd.concat([df_train, df_valid])
    df.reset_index(drop=True, inplace=True)
    print('Number of samples per class:')
    print(df['kfold'].value_counts(dropna=False))
    print('\nClass "-1" is for validation')
    if params.output is None:
        df.to_csv(f'{params.input[:-4]}_folds_groups.csv', index=False)
    else:
        df.to_csv(params.output, index=False)



if __name__ == "__main__":
    main(get_params(sys.argv[1:]))