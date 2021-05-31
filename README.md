# DeepAtlas
This is the repository for the paper "DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation
" at [MICCAI 2019](https://doi.org/10.1007/978-3-030-32245-8_47) [[arxiv]](https://arxiv.org/abs/1904.08465) by Zhenlin Xu and Marc Niethammer.

Install ```torch>=1.0``` and ```torchvision``` according to your config
Install other dependencies with ```pip install -r requirements.txt```

## Data
The [OAI](https://nda.nih.gov/oai/) knee MRIs and [MindBoggle101](https://mindboggle.info/data.html) brain MRIs were used in the paper. We can only share our processed brain MRIs due to copyright limitations. Please download the preprocessed Mindboggle101 from [Google Drive](https://drive.google.com/drive/folders/1UA7sDzIeOvA7niQ3AR6YYNV4UQrfgG_C?usp=sharing). Note that, in DeepAtlas paper, a few images were disregarded due the segmentation labeling errors that were fixed in the later versions of MB101.

## Train a segmentation model with mindboggle data
E.g. Train a segmentation model with 21 training samples
```python train_seg.py --num-samples 21 --data-root $DATA_ROOT --num-epochs 100 --lr 1e-3 --log-root ./logs```
