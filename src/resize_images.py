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
    
    OUTPUT_SIZE = (params.width, params.height)
    # Я бы распаковал как WIDTH, HEIGHT = params.width, params.height
    INPUT_DIR = params.input
    OUTPUT_DIR = params.output
    N_PROCESSES = params.processes
    EXTENTION = params.extention
    # Дело твое, я бы не именовал их так - параметры же
    # задаются динамически. Но это как сам знаешь
    print((f'Resize params: width is {OUTPUT_SIZE[0]}, height is {OUTPUT_SIZE[1]}.'
           f'\nNumber of processes: {N_PROCESSES}.'
           f'\nLooking for .{EXTENTION} images.'))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pathes = glob(os.path.join(INPUT_DIR, f'*.{EXTENTION}'))
    print(f'{len(pathes)} images found at {INPUT_DIR}')
    print(f'Started resizing process, saving resized images to {OUTPUT_DIR}')
    
    def resize_img(img_path: str):
        img = Image.open(img_path)
        img = img.resize(OUTPUT_SIZE, Image.BICUBIC)
        img.save(os.path.join(OUTPUT_DIR, os.path.basename(img_path)))
    # Функция в __main__? Зачем? Не православно

    with Pool(N_PROCESSES) as p:
        list(tqdm(p.imap(resize_img, pathes), total=len(pathes)))
