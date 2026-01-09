# PDCMatch
Probabilistic Deformation Consistency for Unsupervised Shape Matching, AAAI-2026

## Overview
<p align="center">
  <img src="figures/framework.png" width="900">
</p>

## Probabilistic Deformation
<p align="center">
  <img src="figures/illustration.png" width="600">
</p>

## Qualitative Results
<p align="center">
  <img src="figures/qualitative.png" width="900">
</p>
Qualitative comparisons demonstrate the effectiveness of our method under challenging scenarios.

## Installation
```bash 
conda create -n fmnet python=3.8 # create new virtual environment
conda activate fmnet
conda install pytorch cudatoolkit -c pytorch # install pytorch
pip install -r requirements.txt # install other necessary libraries via pip
```


## Dataset
To train and test datasets used in this paper, please download the datasets from the this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) and put all datasets under ../data/
```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for FAUST_r.
```python
python preprocess.py --data_root ../data/FAUST_r/ --no_normalize --n_eig 200
```

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/faust.yaml
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/faust.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## Texture Transfer
An example of texture transfer is provided in *[texture_transfer.py](texture_transfer.py)*
```python
python texture_transfer.py
```
if not useful
```python
xvfb-run -a python visualize.py --opt options/hybrid_ulrssm/test/smal.yaml
```

## Pretrained models
You can find all pre-trained models in [checkpoints](checkpoints) for reproducibility.

If you find this project useful, please cite:

```
@inproceedings{xia2025probabilistic,
  title={Probabilistic Deformation Consistency for Unsupervised Shape Matching},
  author={Xia, Yifan, Ye, Tianwei, Jun Huang, Xiaoguang Mei, Jiayi Ma},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
