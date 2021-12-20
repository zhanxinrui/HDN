

# hdn Training Tutorial

This implements training of hdn.
### Add hdn to your PYTHONPATH
```bash
export PYTHONPATH=/path/to/hdn:$PYTHONPATH
```

## Prepare training dataset
Prepare training dataset, detailed preparations are listed in [training_dataset](training_dataset) directory.
* [COCO](http://cocodataset.org)
* [GOT10K](http://got-10k.aitestunion.com/)
* [POT](https://www3.cs.stonybrook.edu/~hling/data/POT-210/planar_benchmark.html#:~:text=Planar%20object%20tracking%20is%20an,than%20in%20constrained%20laboratory%20environment.)

## Download pretrained backbones
Download pretrained backbones from [here](https://drive.google.com/drive/folders/1DuXVWVYIeynAcvt9uxtkuleV6bs6e3T9) and put them in `project_root/pretrained_models` directory


## Training

### Multi-processing Distributed Data Parallel Training

Refer to [Pytorch distributed training](https://pytorch.org/docs/stable/distributed.html) for detailed description.

#### Single node, multiple GPUs (We use 4 GPUs):

```bash
cd experiments/tracker_homo_config
```
set desired config in [proj_e2e_GOT_unconstrained_v2.yaml](experiments/tracker_homo_config/proj_e2e_GOT_unconstrained_v2.yaml)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=8845 \
../../tools/train.py --cfg proj_e2e_GOT_unconstrained_v2.yaml
```
