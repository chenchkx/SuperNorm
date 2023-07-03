# Improving Expressivity of GNNs with Subgraph-specific Factor Embedded Normalization (KDD'23)

<p align="center">
   <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.10 %20%7C%201.12 %20%7C%202.0-673ab7.svg" alt="Tested PyTorch Versions"></a>
   <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-4caf50.svg" alt="License"></a>
   <a href="https://arxiv.org/abs/2305.19903" target="_blank"><img src="https://img.shields.io/badge/arXiv-2301.12900-009688.svg" alt="arXiv"></a>
</p>

## Overview
Official code for paper [Improving Expressivity of GNNs with Subgraph-specific Factor Embedde Normalization](https://arxiv.org/abs/2211.12712), which proposes a dedicated plug-and-play normalization scheme to strengthen the representative capabilities of GNNs, that is:
<ul>
<li> extending GNNs to be at least as powerful as the 1-WL test in distinguishing non-isomorphism graphs.</li>
<li> alleviating over-smoothing issue while GNNs' layers going deeper.</li>
</ul>


**Citing this repository:** If you find this repository helpful for your research, please kindly cite the following paper:

```
@inproceedings{chen2023improving,
  title={Improving Expressivity of GNNs with Subgraph-specific Factor Embedded Normalization},
  author={Chen, Kaixuan and Liu, Shunyu and Zhu, Tongtian and Qiao, Ji and Su, Yun and Tian, Yingjie and others},
  booktitle={ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2023}
}
```
**Contact:** Please feel free to contact via email (<chenkx@zju.edu.cn>) if you have any questions.

### 1. Environment Installation

```
conda create -n torch python=3.9
conda activate torch
conda install pytorch==1.10.0 torchvision cudatoolkit=11.3 -c pytorch
pip install dgl-cu113==0.7.1 -f https://data.dgl.ai/wheels/repo.html
pip install ogb
pip install seaborn
pip install openpyxl
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
```

### 2. Usage

### 2.1 quick start on ogbg-moltoxcast

### download dataset

```
python download_dataset.py --dataset_name 'ogbg-moltoxcast'
```

### run 4 layers GCN using supernorm and batchnorm

```
python main_graph.py --model 'GCN' --num_layer 4 --norm_type 'supernorm'
python main_graph.py --model 'GCN' --num_layer 4 --norm_type 'batchnorm'
```

### run 16 layers GCN using supernorm and batchnorm

```
python main_graph.py --model 'GCN' --num_layer 16 --norm_type 'supernorm'
python main_graph.py --model 'GCN' --num_layer 16 --norm_type 'batchnorm'
```

### 2.2 Experiments for Graph Isomorphism Test

Firstly, download dataset:

```
python download_dataset.py --dataset_name 'imdb-binary'
```

Then, please find shell files in 'cripts/graph-imdb-sl',  and running

```
sh scripts/graph-imdb-sl/run_gin_sl_supernorm.sh
```

To reproduce the Figure 3 in the paper, please remove the warmup operation (i.e, delete --lr_warmup in shell files.)

### 2.3 Experiments for Over-smoothing Issue

Firstly, download dataset:

```
python download_dataset.py --dataset_name 'cora'
```

Then, please find shell files in 'scripts/node-cora', and running

```
sh scripts/node-cora/run_gcn_supernorm.sh
```

### 2.4 Experiments on other datasets

<!-- ### For example: -->

### 2.4.1 Graph-Level:

   download ogbg-moltoxcast:

```
   python download_dataset.py --dataset_name 'ogbg-moltoxcast'
```

   Then, please find shell files in 'scripts/ogbg-toxcast', and running

```
   sh scripts/ogbg-toxcast/run_gcn_supernorm.sh
```

### 2.4.2 Node-Level:

   download pumbed:

```
   python download_dataset.py --dataset_name 'pubmed'
```

   Then, please find shell files in 'scripts/node-pubmed', and running

```
   sh scripts/node-pubmed/run_gcn_supernorm.sh
```

### 2.4.3 Link-Level:

   download ogbl-collab:

```
   python download_dataset.py --dataset_name 'ogbl-collab'
```

   Then, please find shell files in 'scripts/ogbl-collab', and running

```
   sh scripts/ogbl-collab/run_gcn_supernorm.sh
   sh scripts/ogbl-collab/run_gcn_batchnorm.sh
```


