# HetCAN

Source code for paper "Collaborative Metapath Enhanced Corporate Default Risk Assessment on Heterogeneous Graph".

## Requirements

* python == 3.10.11
* pytorch == 2.0.1
* dgl == 1.1.0
* numpy == 1.24.3
* pandas == 1.1.0
* scikit-learn == 1.2.2
* scipy == 1.10.1

## Usage

* Create a new directory `data/raw` under the project path:

* Download DBLP data from [here](https://cloud.tsinghua.edu.cn/d/2d965d2fc2ee41d09def/files/?p=%2FDBLP.zip&dl=1) and extract the zip file to `data/raw`, where the dataset has already been preprocessed in [Heterogeneous Graph Benchmark(HGB)](https://github.com/THUDM/HGB)
```
cd data/raw
unzip DBLP.zip
```

* Run HetCAN on the public dataset DBLP:
```
python main.py
```
