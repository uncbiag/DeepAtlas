# DeepAtlas
This is the repository for the paper "DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation
" at [MICCAI 2019](https://doi.org/10.1007/978-3-030-32245-8_47) [[arxiv]](https://arxiv.org/abs/1904.08465) by Zhenlin Xu and Marc Niethammer.

Install ```torch>=1.0``` and ```torchvision``` according to your config
Install other dependencies with ```pip install -r requirements.txt```

## Train a segmentation model with mindboggle data
E.g. Train a segmentation model with 21 training samples
```python train_seg.py --num-samples 21 --data-root $DATA_ROOT --num-epochs 100 --lr 1e-3 --log-root ./logs```