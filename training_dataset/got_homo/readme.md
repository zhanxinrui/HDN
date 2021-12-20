# Preprocessing GOT10K (train and val)


### Crop & Generate data info (20 min)

````shell
rm ./train/list.txt
rm ./val/list.txt

python mkdir_got_train.py
python parse_got10k.py
python par_crop.py 511 32
python gen_json.py
````
