import argparse
import os
import sys
from PIL import Image
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
# Отсортировать зависимости


def get_params(args):
    parser = argparse.ArgumentParser(description='Resizes your dataset.\n***CAUTION: -h is for help, while -H is for height***')
    parser.add_argument('-w', '--width', help='Width of resized images.', 
                        dest='width', type=int)
    parser.add_argument('-H','--height', help='Height of resized images.', 
                        dest='height', type=int)
                        # а где дефолтные значения?
    parser.add_argument('-p', '--processes', help='Number of processes to be used.', 
                        dest='processes', type=int, default=8)
    parser.add_argument('-x', '--extention', help='Extention of images.', 
                        dest='extention', default='jpg')
    parser.add_argument('-i', '--input', help='Path to images.', 
                        dest='input', default='../input/images/')
    parser.add_argument('-o','--output', help='Path to resized images.', 
                        dest='output', default='../input/images_resized/')
    return parser.parse_args(args)


if __name__ == "__main__":
    params = get_params(sys.argv[1:])
    print((f'Resize params: width is {params.width}, height is {params.height}.'
           f'\nNumber of processes: {params.processes}.'
           f'\nLooking for .{params.extention} images.'))

    os.makedirs(params.output, exist_ok=True)
    pathes = glob(os.path.join(params.input, f'*.{params.extention}'))
    print(f'{len(pathes)} images found at {params.input}')
    print(f'Started resizing process, saving resized images to {params.output}')
    
    def resize_img(img_path: str):
        img = Image.open(img_path)
        img = img.resize((params.width, params.height), Image.BICUBIC)
        img.save(os.path.join(params.output, os.path.basename(img_path)))


    with Pool(params.processes) as p:
        list(tqdm(p.imap(resize_img, pathes), total=len(pathes)))
