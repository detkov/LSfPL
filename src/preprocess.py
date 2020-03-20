import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

OUTPUT_SIZE = (512, 340)
OUTPUT_DIR = '../input/images_res/'

if __name__ == "__main__":
    # remove duplicates
    train = pd.read_csv('../input/train.csv')
    train.drop(index=1173, inplace=True)
    train.to_csv('../input/train.csv', index=False)
    
    # resize and rotate images in "portrait" mode
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    
    pathes = glob('../input/images/*.jpg')
    for path in tqdm(pathes):
        image = Image.open(path)
        if image.height > image.width:
            image = image.transpose(Image.ROTATE_90)
        image = image.resize(OUTPUT_SIZE, Image.BICUBIC)
        image.save(OUTPUT_DIR + os.path.basename(path))