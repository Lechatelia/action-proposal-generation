# BSN-boundary-sensitive-network.pytorch

This repo holds the pytorch-version codes of paper: "BSN: Boundary Sensitive Network for Temporal Action Proposal Generation", which is accepted in ECCV 2018. You can also find tensorflow-version implementation in [[BSN.tensorflow]](https://github.com/wzmsltw/BSN-boundary-sensitive-network).

[[Arxiv Preprint]](http://arxiv.org/abs/1806.02964)

# Update
support python3


# old comit 
* 2018.12.12: Release Pytorch-version BSN
* 2018.09.26: Previously, we adopted classification results from result files of "Uts at activitynet 2016" for action detection experiments. Recently we found that the classification accuracy of these results are unexpected high. Thus we replace it with classification results of "cuhk & ethz & siat submission to activitynet challenge 2017" and updated all related experiments accordingly. You can find updated papers in my [homepage](wzmsltw.github.io) and in arXiv.
* 2018.07.09: Codes and feature of BSN
* 2018.07.02: Repository for BSN



# Contents

* [Paper Introduction](#paper-introduction)
* [Other Info](#other-info)
* [Prerequisites](#prerequisites)
* [Code and Data Preparation](#Code_and_Data_Preparation)
* [Training and Testing  of BSN](#Training_and_Testing_of_BSN)

# Paper Introduction

 <img src="./paper_pic/eccv_overview.jpg" width = "700" alt="image" align=center />

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts “local to global” fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.


# Prerequisites

These code is  implemented in Pytorch 0.4.1 + Python2 + tensorboardX. Thus please install Pytorch first.

# Code and Data Preparation

## Get the code

Clone this repo with git, please use:

```
git clone https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch.git
```



## Download Datasets

We support experiments with publicly available dataset ActivityNet 1.3 for temporal action proposal generation now. To download this dataset, please use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube.

To extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, which is the challenge solution of CUHK&ETH&SIAT team in ActivityNet challenge 2016. Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF/view?usp=sharing) or [Baidu Yun](). If you download features using BaiduYun, please use `cat zip_csv_mean_100.z* > csv_mean_100.zip` before unzip. After download and unzip, please put `csv_mean_100` directory to `./data/activitynet_feature_cuhk/` .

## About the rescaled feature supportted by the author 
In the [video_info.csv](data/activitynet_annotations/video_info_new.csv), there are 19228 videos.
However, when you use the  `cat zip_csv_mean_100.z* > csv_mean_100.zip` and `unzip csv_mean_100.zip` on ubuntu, there may be only 3k+ video feature csv.
By zipping this zip on windows, it seems ok.

I find you can use the following attempts to get the 19228 video feature csv files:
`zip -FF  zip_csv_mean_100.zip --out csv_mean_100.zip
unzip -FF csv_mean_100.zip`.
This works for me on Ubuntu 16.04

# Training and Testing  of BSN

All configurations of BSN are saved in opts.py, where you can modify training and model parameter.


#### 1. Training of temporal evaluation module


```
python main.py --module TEM --mode train
```

We also provide trained TEM model in `./checkpoint/`

#### 2. Testing of temporal evaluation module

```
python main.py --module TEM --mode inference
```

#### 3. Proposals generation and BSP feature generation

```
python main.py --module PGM
```

#### 4. Training of proposal evaluation module

```
python main.py --module PEM --mode train
```

We also provide trained PEM model in `./checkpoint` .

#### 6. Testing of proposal evaluation module

```
python main.py --module PEM --mode inference --pem_batch_size 1
```

#### 7. Post processing and generate final results

```
python main.py --module Post_processing
```

#### 8. Eval the performance of proposals

```
python main.py --module Evaluation
```

You can find evaluation figure in `./output`

You can also simply run all above commands using:

```
sh bsn.sh
```

# Other Info

## Citation


Please cite the following paper if you feel SSN useful to your research

```
@inproceedings{BSN2018arXiv,
  author    = {Tianwei Lin and
               Xu Zhao and
               Haisheng Su and
               Chongjing Wang and
               Ming Yang},
  title     = {BSN: Boundary Sensitive Network for Temporal Action Proposal Generation},
  booktitle   = {European Conference on Computer Vision},
  year      = {2018},
}
```


## Contact
For any question, please file an issue or contact
```
Tianwei Lin: wzmsltw@sjtu.edu.cn
```
