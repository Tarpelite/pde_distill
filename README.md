# pde_distill
Distill the PDE dataset


## Data

The Data could be downloaded from [1DCFD](https://darus.uni-stuttgart.de/file.xhtml?fileId=164672&version=8.0)


## Run Buffer 
```bash
python buffer.py +args=train_1DCFD.yaml
```

## Run Distill 
```bash
CUDA_VISIBLE_DEVICES=0 python distill.py +args=distill_1DCFD.yaml
```