# PySlowFast

PySlowFast is an open source video understanding codebase from FAIR that provides state-of-the-art video classification models with efficient training. This repository includes implementations of the following methods:

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [A Multigrid Method for Efficiently Training Video Models](https://arxiv.org/abs/1912.00998)
- [X3D: Progressive Network Expansion for Efficient Video Recognition](https://arxiv.org/abs/2004.04730)

<div align="center">
  <img src="demo/ava_demo.gif" width="600px"/>
</div>

## Introduction

The goal of PySlowFast is to provide a high-performance, light-weight pytorch codebase provides state-of-the-art video backbones for video understanding research on different tasks (classification, detection, and etc). It is designed in order to support rapid implementation and evaluation of novel video research ideas. PySlowFast includes implementations of the following backbone network architectures:

- SlowFast
- Slow
- C2D
- I3D
- Non-local Network
- X3D

## Updates

- We now support [X3D Models](https://arxiv.org/abs/2004.04730). See [`projects/x3d`](./projects/x3d/README.md) for more information.
- We now support [Multigrid Training](https://arxiv.org/abs/1912.00998) for efficiently training video models. See [`projects/multigrid`](./projects/multigrid/README.md) for more information.
- PySlowFast is released in conjunction with our [ICCV 2019 Tutorial](https://alexander-kirillov.github.io/tutorials/visual-recognition-iccv19/).

## License

PySlowFast is released under the [Apache 2.0 license](LICENSE).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the PySlowFast [Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for PyTorch and PySlowFast in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](slowfast/datasets/DATASET.md) to prepare the datasets.

## Quick Start

Follow the example in [GETTING_STARTED.md](GETTING_STARTED.md) to start playing video models with PySlowFast.

## Visualization Tools

We offer a range of visualization tools for the train/eval/test processes, model analysis, and for running inference with trained model.
More information at [Visualization Tools](VISUALIZATION_TOOLS.md).

## Contributors

PySlowFast is written and maintained by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).

## Citing PySlowFast

If you find PySlowFast useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

## ???????????????

???1???/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/config/defaults.py
???2???/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/utils/parser.py
???3???/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1.yaml

???????????????/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/utils/misc.py ---???get_class_names

## faster rcnn ?????????

????????? clip ???????????? 32 ????????????????????????????????????
/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/detectron2/detectron2/engine/defaults.py

## slowfast ????????????

/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/predictor.py

## ??????????????? buffer--> task

/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/demo_loader.py
/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/utils.py

## trace ???????????????

???1???/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/models/video_model_builder.py
???2???/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/predictor.py

# ???????????????????????????

/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/async_predictor.py
????????????/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/visualization/video_visualizer.py

#?????? demo
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/config/defaults.py

# train 5->4

/home/window_share/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/utils/misc.py 153-157
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/models/video_model_builder.py 391-392
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/tools/train_net.py   85-88

?????????????????????
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/datasets/ava_helper.py

# slowfast  train:
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/tools/train_net.py
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/utils/misc.py

# X3D:
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/config/defaults.py  449       --->    _C.DETECTION.SPATIAL_SCALE_FACTOR = 32    ## 16-->slowfast,32-->X3D
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/utils/misc.py       154
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/tools/train_net.py        84
/home/os/window_share/ganhaiyang/Alg_Proj/Recog_Proj/SlowFast/slowfast/models/video_model_builder.py  748->789

slowfast:
  (head): ResNetRoIHead(
    (s0_tpool): AvgPool3d(kernel_size=[8, 1, 1], stride=1, padding=0)
    (s0_roi): ROIAlign(output_size=[7, 7], spatial_scale=0.0625, sampling_ratio=0, aligned=True)
    (s0_spool): MaxPool2d(kernel_size=[7, 7], stride=1, padding=0, dilation=1, ceil_mode=False)
    (s1_tpool): AvgPool3d(kernel_size=[32, 1, 1], stride=1, padding=0)
    (s1_roi): ROIAlign(output_size=[7, 7], spatial_scale=0.0625, sampling_ratio=0, aligned=True)
    (s1_spool): MaxPool2d(kernel_size=[7, 7], stride=1, padding=0, dilation=1, ceil_mode=False)
    (dropout): Dropout(p=0.5, inplace=False)
    (projection): Linear(in_features=2304, out_features=80, bias=True)
    (act): Sigmoid()
  )