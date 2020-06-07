python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/jpeg/test/ -o ../input/images_resized/
python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/jpeg/train/ -o ../input/images_resized/
python create_folds.py -v 10 -f image_name 
