# EIC-MTP

This repository contains the official code from ''Explicit-Implicit Combined Learning for Multi-Modal Trajectory Prediction in Multi-Agent Systems''

## Installation

1. Clone this repository 

2. Set up a new conda environment 
``` shell
conda create --name mtp python=3.7
```

3. Install dependencies
```shell
conda activate mtp
pip install nuscenes-devkit
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0
pip install ray
pip install psutil
pip install positional-encodings==5.0.0
pip install imageio==2.23.0
pip install tensorboard

## Dataset

1. Download the [nuScenes dataset](https://www.nuscenes.org/download). For this project we just need the following.
    - Metadata for the Trainval split (v1.0)
    - Map expansion pack (v1.3)

2. Organize the nuScenes root directory as follows
```plain
└── nuscenes/
    ├── maps/
    |   ├── basemaps/
    |   ├── expansion/
    |   ├── prediction/
    |   ├── 36092f0b03a857c6a3403e25b4b7aab3.png
    |   ├── 37819e65e09e5547b8a3ceaefba56bb2.png
    |   ├── 53992ee3023e5494b90c316c183be829.png
    |   └── 93406b464a165eaba6d9de76ca09f5da.png
    └── v1.0-trainval
        ├── attribute.json
        ├── calibrated_sensor.json
        ...
        └── visibility.json         
```

3. Run the following script to extract pre-processed data. This speeds up training significantly.
```shell
python preprocess.py -c configs/preprocess_nuscenes.yml -r path/to/dataset/nuScenes/ -d ./preprocessed/
```

## Inference
```shell
python evaluate.py -c configs/motif_v64.yml -r path/to/dataset/nuscenes/ -d ./preprocessed/ -o ./output/motif_v64/ -w ./output/motif_v64/checkpoints/best.tar
```
## Training
```shell
python train.py -c configs/motif_v64.yml -r path/to/dataset/nuscenes/ -d ./preprocessed/ -o ./output/motif_v64/ -n 500
```

To launch tensorboard, run
```shell
tensorboard --logdir=path/to/output/directory/tensorboard_logs
```
