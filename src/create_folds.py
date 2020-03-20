import argparse
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import sys

FEATURES = 'image_id'

def get_params(args):
    parser = argparse.ArgumentParser(description='Makes a split of your train file.')
    parser.add_argument('-i', '--input', help='Path to `train.csv`.', 
                        dest='input', default='../input/train.csv')
    parser.add_argument('-o','--output', help='Path to `tarin.csv` with folds.', 
                        dest='output', default='../input/train_folds.csv')
    parser.add_argument('-s', '--splits', help='Number of splits.', 
                        dest='splits', type=int, default=5)
    parser.add_argument('-r', '--randomstate', help='random_state parameter value.', 
                        dest='randomstate', type=int, default=42)
    return parser.parse_args(args)


if __name__ == "__main__":
    params = get_params(sys.argv[1:])
    df = pd.read_csv(params.input)
    df['kfold'] = np.nan
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[FEATURES].values
    y = df.iloc[:, 2] + df.iloc[:, 3] * 2 + df.iloc[:, 4] * 3

    skf = StratifiedKFold(n_splits=params.splits, random_state=params.randomstate)
    for fold, (train, valid) in enumerate(skf.split(X, y)):
        df.loc[valid, 'kfold'] = fold

    print(df['kfold'].value_counts(dropna=False))
    print(df.head())
    df.to_csv(params.output, index=False)