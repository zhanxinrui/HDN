# Preprocessing GOT10K (train and val)


### Crop & Generate data info (20 min)
download GOT10K in your cfg.BASE.DATA_PATH
````shell
#ln -s /your_GOT_data_path/ /path/to/data_path/GOT10k
#mkdir /path/to/data_path/GOT10k
python mkdir_got_train.py
python parse_got10k.py
python par_crop.py 511 32
python gen_json.py
````
