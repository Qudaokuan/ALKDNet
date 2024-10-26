# [Asymmetric Large Kernel Distillation Network for Efficient Single Image Super-Resolution](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1502499/abstract)

Daokuan Qu and Yuyao Ke

## Table of Contents

1. [Introduction](#introduction)
2. [Results](#introduction)
3. [Preparation](#preparation)
4. [Testing](#testing)
5. [Training](#training)
6. [Citation](#citation)

## Introduction

Recently, significant advancements have been made in the field of efficient single-image super-resolution, primarily driven by the innovative concept of information distillation. This method adeptly leverages multi-level features to facilitate high-resolution image reconstruction, allowing for enhanced detail and clarity. However, many existing approaches predominantly emphasize the enhancement of distilled features, often overlooking the critical aspect of improving the feature extraction capabilities of the distillation module itself.
In this paper, we address this limitation by introducing an asymmetric large-kernel convolution design. By increasing the size of the convolution kernel, we expand the receptive field, which enables the model to more effectively capture long-range dependencies among image pixels. This enhancement significantly improves the model's perceptual ability, leading to more accurate reconstructions.
To maintain a manageable level of model complexity, we adopt a lightweight architecture that employs asymmetric convolution techniques. Building on this foundation, we propose the Lightweight Asymmetric Large Kernel Distillation Network (ALKDNet). Comprehensive experiments reveal that ALKDNet not only preserves efficiency but also achieves state-of-the-art performance compared to existing super-resolution methods. 

<img src=./assets/ALKDN.jpg />

## Results

<img src=./assets/Results.png />

## preparation
### Environment

[PyTorch >= 1.7](https://pytorch.org/)  
[BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

### Installation
```
pip install -r requirements.txt
python setup.py develop
```

## Testing
· Refer to ./options/test for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
· The pretrained models are available in ./experiments/pretrained_models/  
· Then run the follwing codes:  

```
python basicsr/test.py -opt options/test/test_ALKDN_x2.yml
python basicsr/test.py -opt options/test/test_ALKDN_x3.yml
python basicsr/test.py -opt options/test/test_ALKDN_x4.yml
```
The testing results will be saved in the ./results folder.

## Training
· Refer to ./options/train for the configuration file of the model to train.  
· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  
· The training command is like  
```
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_ALKDN_x2.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_ALKDN_x3.yml
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/train_ALKDN_x4.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  

## Citation

If you find the code useful in your research, please cite:

    @article{qu18asymmetric,
      title={Asymmetric Large Kernel Distillation Network for Efficient Single Image Super-Resolution},
      author={Qu, Daokuan and Ke, Yuyao},
      journal={Frontiers in Neuroscience},
      volume={18},
      pages={1502499},
      publisher={Frontiers}
    }

## License

See [MIT License](/LICENSE)
