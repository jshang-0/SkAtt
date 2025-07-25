# Sketching-Attentions-with-Lower-Linear-Complexity-for-Long-Sequence-Problems
This repository is the implementation for the paper "Sketching Attentions with Lower Linear Complexity for Long Sequence Problems"


## Requirements

To install requirements in a conda environment:

```
pip install -r requirements.txt
```

Note: Specific requirements for data preprocessing are not included here.


# Data Preparation

Processed files can be downloaded [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing), or processed with the following steps:

1. Requirements

```
tensorboard>=2.3.0
tensorflow>=2.3.1
tensorflow-datasets>=4.0.1
```

2. Download [the TFDS files for pathfinder](https://storage.cloud.google.com/long-range-arena/pathfinder_tfds.gz) and then set _PATHFINER_TFDS_PATH to the unzipped directory (following https://github.com/google-research/long-range-arena/issues/11)
3. Download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) (7.7 GB).
4. Unzip `lra-release` and put under `./data/`.

```
cd data
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
tar zxvf lra-release.gz 
```

5. Create a directory `lra_processed` under `./src/data/`.

```
mkdir lra_processed
cd ..
```

6.The directory structure would be (assuming the root dir is `src`)

```
./data/lra-processed
./data/long-range-arena-main
./data/lra_release
```

7. Create train, dev, and test dataset pickle files for each task.

```
cd preprocess
python create_pathfinder.py
python create_listops.py
python create_retrieval.py
python create_text.py
python create_cifar10.py
```

Note: most source code comes from [LRA repo](https://github.com/google-research/long-range-arena).



# Run 

Create a folder to save checkpoints under `src` folder: 

```
cd src
mkdir checkpoints
```

Modify the configuration in `config.py` and run under `src` folder:

```
python main.py --mode train --attn sketch_exp_kernel --task lra-text
```

- mode: `train`, `eval`
- attn: `softmax`, `linformer`, `performer`, `exp_kernel`, `sketch_exp_kernel`, `rsketch_exp_kernel`
- task: `lra-listops`, `lra-pathfinder`, `lra-retrieval`, `lra-text`, `lra-image`


Note: If your data is not located in `./src/data/`, create a softlink with `ln -s <datapath> ./src/data`

