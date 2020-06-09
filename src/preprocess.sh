python resize_images.py -w 256 -H 256 -p 8 -x jpg -i ../input/jpeg/test/ -o ../input/images_resized_256x256/
python resize_images.py -w 256 -H 256 -p 8 -x jpg -i ../input/jpeg/train/ -o ../input/images_resized_256x256/
python create_folds.py -v 10 -f image_name 
