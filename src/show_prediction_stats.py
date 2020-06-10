import argparse
import sys
from os.path import join
from typing import List, Union, Tuple
import pandas as pd
import numpy as np


def get_params(args):
    parser = argparse.ArgumentParser(description='Shows statistics on target predictions.')
    parser.add_argument('-d', '--file_dir', help='File directory.', 
                        dest='file_dir', default='../submissions')
    parser.add_argument('-f', '--file_name', help='File name.', 
                        dest='file_name', required=True)
    parser.add_argument('-t', '--target', help='Target feature.', 
                        dest='target', default='target')
    parser.add_argument('-n', '--df_name', help='Name of the dataset.', 
                        dest='df_name', default='test')
    return parser.parse_args(args)


custom_bins = [[0, [0, 0.05]], [0, [0, 0.10]], [0, [0, 0.15]], [0, [0, 0.20]], [0, [0, 0.50 - 1e-12]], 
               [1, [0.50, 1]], [1, [0.80, 1]], [1, [0.85, 1]], [1, [0.90, 1]], [1, [0.95, 1]]]


def print_distribution_info(df: pd.DataFrame, target: str, name_of_df: str):
    print(f'Number of samples in {name_of_df}: {df.shape[0]}')
    print(f'Target dictribution:')
    print(df[target].apply(round).value_counts(normalize=True) * 100)
    print()


def print_bins_distribution_info(df: pd.DataFrame, 
                                 custom_bins: List[Tuple[int, Tuple[float, float]]] = custom_bins):
    rows = []
    class_0_df, class_1_df = df[df['target'] < 0.50], df[df['target'] >= 0.50]
    
    for i, (class_, custom_bin) in enumerate(custom_bins):
        class_df_len = class_0_df.shape[0] if class_ == 0 else class_1_df.shape[0]
        
        hist, _ = np.histogram(df['target'], bins=custom_bin)
        
        abs_val = hist[0]
        pct_val = round(abs_val / df.shape[0] * 100, 2)        
        pct_class_val = round(abs_val / class_df_len * 100, 2)
        
        rows.append([abs_val, f'{pct_val:.2f}', f'{pct_class_val:.2f}'])

    indices = [str(custom_bin) for (class_, custom_bin) in custom_bins]
    indices[len(indices)//2-1] = '[0, 0.5)'

    rows_df = pd.DataFrame(rows, index=indices, columns=['Abs', '%', '% of its class'])

    print('Number (N) of predictions in bins (Assuming class = 0 if p<0.5 else 1):')
    print(rows_df)


def main(params):
    df = pd.read_csv(join(params.file_dir, params.file_name))
    print_distribution_info(df, params.target, params.df_name)
    print_bins_distribution_info(df)

if __name__ == "__main__":
    main(get_params(sys.argv[1:]))