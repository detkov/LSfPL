python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/test/ -o ../input/test_resized/
python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/train/ -o ../input/train_resized/
python create_folds.py -v 10 -f image_name 