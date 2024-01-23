# HetCAN

Source code for paper "Collaborative Metapath Enhanced Corporate Default Risk Assessment on Heterogeneous Graph".

## Requirements

* python == 3.10.11
* pytorch == 2.0.1
* dgl == 1.1.0
* numpy == 1.24.1
* pandas == 2.1.1
* scikit-learn == 1.3.1
* scipy == 1.11.3

## Usage

* Create a new directory `data/` under the project path:

* Download DBLP data from [here](https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX?usp=sharing) and extract the zip file to `data/`, where the dataset has already been preprocessed in [Heterogeneous Graph Benchmark(HGB)](https://github.com/THUDM/HGB)
```
cd data/
unzip DBLP.zip
```

* Run HetCAN on the public dataset DBLP:
```
python main.py
```
