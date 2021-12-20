# Installation

This document contains detailed instructions for installing dependencies for hdn. The code is tested on an Ubuntu 18.04 system with Nvidia GPU (2080Ti).



### Requirments
* Conda with Python 3.7
* Nvidia GPU
* PyTorch >= 1.3.1
* pyyaml
* yacs
* tqdm
* matplotlib
* OpenCV
* ....

## Step-by-step instructions
#### Clone our project 
```bash
git clone https://github.com/zhanxinrui/HDN.git
```

#### Create environment and activate

```bash
conda create --name hdn python=3.7
source activate hdn
```

#### Install numpy/pytorch/opencv
```bash
conda install numpy==1.17.0

conda install pytorch=1.3.1 torchvision cudatoolkit=10.1 -c pytorch
or install torch through pip: 
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

pip install opencv-python opencv-contrib-python
```

#### Install other requirements
```bash
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboard future  optuna graphviz scipy imageio kornia==0.2.1 Pillow optuna psutil memory_profiler shapely orjson mpi4py
```

#### Build extensions
```bash
cd HDN
python setup.py build_ext --inplace
```


