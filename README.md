# HDN

This project is the offical PyTorch implementation of HDN(Homography Decomposition Networks) for planar object tracking.

### [Project Page](https://zhanxinrui.github.io/HDN-homepage/) | [Paper](https://arxiv.org/abs/2112.07909)

```
@misc{zhan2021homography,
      title={Homography Decomposition Networks for Planar Object Tracking}, 
      author={Xinrui Zhan and Yueran Liu and Jianke Zhu and Yang Li},
      year={2021},
      eprint={2112.07909},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<div align="left">
  <img src="./demo/output/demo.gif" width="350px" />
</div>

## Installation

Please find installation instructions in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using HDN

### Add HDN to your PYTHONPATH
```bash
vim ~/.bashrc
# add home of project to PYTHONPATH
export PYTHONPATH=/path/to/HDN:/path/to/HDN/homo_estimator/Deep_homography/Oneline_DLTv1:$PYTHONPATH
```

### Download models

[Google Drive](https://drive.google.com/file/d/1eakDIJ8m4cZNaiJvWdKAYHyZt0hv2mAY/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1z4B5oVDgDloTrrXNQn1E6w) (key: 8uhq)

### Base Setting
The global parameters setting file is hdn/core/config.py
You first need to set the base path:

```bash
__C.BASE.PROJ_PATH = /xxx/xxx/project_root/ #/home/Kay/SOT/server_86/HDN/   (path_to_hdn)
__C.BASE.BASE_PATH = /xxx/xxx/ #/home/Kay/SOT/                  (base_path_to_workspace)
__C.BASE.DATA_PATH = /xxx/xxx/data/POT  #/home/Kay/data/POT     (path to POT datasets)
__C.BASE.DATA_ROOT = /xxx/xxx   #/home/Kay/Data/Dataset/        (path to other datasets)
```

### Demo
**Planar Object Tracking and its applications**
we provide 4 modes: 
* tracking: tracking planar object with not less than 4 points in the object.
* img_replace: replacing planar object with image .
* video_replace: replacing planar object with video. 
* mosiac: adding mosiac to planar object.

```bash
python tools/demo.py 
--snapshot model/hdn-simi-sup-hm-unsup.pth 
--config experiments/tracker_homo_config/proj_e2e_GOT_unconstrained_v2.yaml 
--video demo/door.mp4 
--mode img_replace 
--img_insert demo/coke2.jpg #required in mode 'img_replace'  
--video_insert demo/t5_videos/replace-video/   #required in mode 'video_replace'
--save # whether save the results.
```
e.g.
```bash
python tools/demo.py  --snapshot model/hdn-simi-sup-hm-unsup.pth  --config experiments/tracker_homo_config/proj_e2e_GOT_unconstrained_v2.yaml --video demo/door.mp4 --mode img_replace --img_insert demo/coke2.jpg --save
```
we provide some real-world videos [here](https://www.aliyundrive.com/s/ycDqPLz5e3Z)


### Download testing datasets
**POT** 

For [POT](https://liangp.github.io/data/pot280/) dataset, download the videos from [POT280](https://pan.baidu.com/s/1boKIoXGFOWZ-uu9X6WzDCA.) and annotations from 
[here](https://liangp.github.io/data/pot280/annotation.zip)


```bash
1. unzip POT_v.zip and POT_annotation.zip and put them in your cfg.BASE.DATA_PATH #unzip the zip files
  cd POT_v
  unzip "*.zip"
  cd ..

2. mkdir POT
   mkdir path_to_hdn/testing_dataset
   python path_to_hdn/toolkit/benchmarks/POT/pot_video_to_pic.py #video to images  
   ln -s path_to_data/POT  path_to_hdn/testing_dataset/POT #link to testing_datasets


4. python path_to_hdn/toolkit/benchmarks/POT/generate_json_for_POT.py --dataset POT210 #generate json annotation for POT
   python path_to_hdn/toolkit/benchmarks/POT/generate_json_for_POT.py --dataset POT280 

```
**UCSB & POIC**

Download from [here](http://webdocs.cs.ualberta.ca/~vis/mtf/) 
put them in your cfg.BASE.DATA_PATH
```bash
ln -s path_to_data/UCSB  path_to_hdn/testing_dataset/UCSB #link to testing_datasets
```

generate json: 
```bash
  python path_to_hdn/toolkit/benchmarks/POIC/generate_json_for_poic.py #generate json annotation for POT
  python path_to_hdn/toolkit/benchmarks/UCSB/generate_json_for_ucsb.py #generate json annotation for POT
```

**Other datsets**:

Download datasets and put them into `testing_dataset` directory. Jsons of commonly used datasets can be downloaded from [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI). If you want to test tracker on new dataset, please refer to [pysot-toolkit](https://github.com/StrangerZhang/pysot-toolkit) to setting `testing_dataset`.





### Test tracker

- test POT
```bash
cd experiments/tracker_homo_config
python -u ../../tools/test.py \
	--snapshot ../../model/hdn-simi-sup-hm-unsup.pth \ # model path 
	--dataset POT210 \ # dataset name
	--config proj_e2e_GOT_unconstrained_v2.yaml # config file
	--vis   #display video
```

The testing results will in the current directory(./results/dataset/model_name/)


### Eval tracker

#### For POT evaluation

1.use tools/change_pot_results_name.py to convert result_name(you need to set the path in the file).

2.use tools/convert2Homography.py to generate the homo file(you need to set the corresponding path in the file).

3.use POT toolkit to test the results. My version toolkit can be found [here](https://github.com/zhanxinrui/POT_evaluation_toolkit)
or [official](https://drive.google.com/file/d/1oRbi4p-PFqKPOt4SvKVJP0wkGfb1ZR9b/view?usp=sharing)
for other trackers:

#### For others:
For POIC, UCSB or POT evaluation on centroid precision, success rate, and robustness etc.
assuming still in experiments/tracker_homo_config
``` bash
python ../../tools/eval.py 	 \
	--tracker_path ./results \ # result path
	--dataset POIC        \ # dataset name
	--num 1 		 \ # number thread to eval
	--tracker_prefix 'model'   # tracker_name
```

The raw results can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1xJTBITgMyvfUmeZqdzA5GX_ZMMEgLwbt?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1A6CcOBqyD3FU3illbNew6Q) (key:d98h)
###  Training :wrench:
We use the COCO14 and GOT10K as our traning datasets.
See [TRAIN.md](TRAIN.md) for detailed instruction.


## Acknowledgement
This work is supported by the National Natural Science Foundation of China under Grants (61831015 and 62102152) and sponsored by CAAI-Huawei MindSpore Open Fund. 

Our codes is based on [SiamBAN](https://github.com/hqucv/siamban) and [DeepHomography](https://github.com/JirongZhang/DeepHomography).

## License

This project is released under the Apache 2.0 license. 