# RelativeConUCB

This repository contains Pytorch implementation of our paper:

> Zhihui Xie, Tong Yu, Canzhe Zhao, Shuai Li. *Comparison-based Conversational Recommender System with Relative Bandit Feedback*, SIGIR 2021.

## Dependency
We develop using Python 3.7.6 and Pytorch 1.5.1. To create datasets used in the experiments, [LIBMF](https://github.com/cjlin1/libmf) is needed. Datasets can also be downloaded via [Google Drive](https://drive.google.com/file/d/10sMsSHTa5ftyWi0ZPloB6ctYYsiEf3YW/view?usp=sharing).

## Notes
- 2021/12/14: We found a bug related to difference-type algorithms, which has a negative impact on the final performance. It is now fixed.

## Citation
If you use this code for your research, please cite our work:

```
@inproceedings{xie2021comparison,
  title={Comparison-based Conversational Recommender System with Relative Bandit Feedback},
  author={Xie, Zhihui and Yu, Tong and Zhao, Canzhe and Li, Shuai},
  booktitle={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1400--1409},
  year={2021}
}
```