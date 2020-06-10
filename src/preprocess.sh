python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/jpeg/test/ -o ../input/images_resized_512x512/
python resize_images.py -w 512 -H 512 -p 8 -x jpg -i ../input/jpeg/train/ -o ../input/images_resized_512x512/
python create_folds_stratified.py -v 10 -f image_name 
python create_folds_groups.py -v 10 -f image_name -f patient_id