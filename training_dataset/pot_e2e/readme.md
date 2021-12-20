# Preprocessing POT (train and val)


### Crop & Generate data info (20 min)

````shell
python mkdir_pot_train.py
ln data -> cur path
python parse_pot.py
python par_crop.py 511 32
python create_pot_test_list.py
python gen_json.py
````
