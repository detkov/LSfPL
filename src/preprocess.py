import pandas as pd


if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')

    # remove duplicates
    train.drop(index=1173, inplace=True)
    train.to_csv('../input/train.csv', index=False)