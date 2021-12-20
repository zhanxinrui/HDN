# Preprocessing COCO

### Download raw images and annotations 

````shell
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip


unzip ./train2014.zip
unzip ./val2014.zip
unzip ./annotations_trainval2014.zip
cd pycocotools && make && cd ..
ln -s /your_coco_data_path/* ./

````

### Crop & Generate data info (10 min)

````shell
#python par_crop.py [crop_size] [num_threads]
python par_crop.py 511 12
python gen_json.py
````
