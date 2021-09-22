# Contrastive Adaptation Network

This repo contains a modified version of [Contrastive Adaptation Network (CAN)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kang_Contrastive_Adaptation_Network_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.pdf) for time series datasets. We replaced the image feature extractor (ResNet) with the time-series compatible feature extractor from CoDATS. This allows a comparison of CAN with our time series contrastive domain adaptation method [CALDA](https://github.com/floft/calda).

## Dependencies
Below we list which packages and versions we used, though likely the exact versions are not required:

- Python 3.7.4
- PyTorch 1.9.0
- PyYAML 5.3.1
- torchvision 0.10.0
- torchinfo 1.5.3
- easydict 1.9
- pickle5 0.0.11

## Datasets

For time series datasets, see [CALDA](https://github.com/floft/calda) instructions. That repository contains the scripts to generate the pickle files used by this code. Note CALDA should be cloned in `../calda`, i.e. in the parent directory of this repo.

## Training

For the time series training:
```
time ./experiments/scripts/train.sh ./experiments/config/timeseries/CAN/timeseries_train_train2val_cfg.yaml 0 CAN timeseries_train2val
```

The experiment log file and the saved checkpoints will be stored at ./experiments/ckpt/${experiment_name}

## Test

For the time series best-target evaluation (i.e. using a comparable model selection methodology as CALDA):
```
time ./experiments/scripts/test_best_target.sh ./experiments/config/timeseries/timeseries_test_val_cfg.yaml 0 True timeseries_train2val timeseries_test ./experiments/ckpt
```

## Citing
Please cite their paper if you use their code in your research:
```
@article{kangcontrastive,
  title={Contrastive Adaptation Network for Single-and Multi-Source Domain Adaptation},
  author={Kang, Guoliang and Jiang, Lu and Wei, Yunchao and Yang, Yi and Hauptmann, Alexander G},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2020}
}

@inproceedings{kang2019contrastive,
  title={Contrastive Adaptation Network for Unsupervised Domain Adaptation},
  author={Kang, Guoliang and Jiang, Lu and Yang, Yi and Hauptmann, Alexander G},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4893--4902},
  year={2019}
}
```

## Thanks to third party
The way of setting configurations is inspired by <https://github.com/rbgirshick/py-faster-rcnn>.
