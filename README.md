Introduction
------
Code for paper "Collaborative Metapath Enhanced Corporate Default Risk Assessment on Heterogeneous Graph", which is accepted by WWW 2024.

HetCAN is developped for small company default risk assessment based on heterogeneous graph and collaborative metapath. It is designed for pre-loan services for small businesses and real-world banking data.

Main Contact: 

- Zheng Zhang (zhang.zh0707@gmail.com)
- Yingsheng Ji (jiyingsheng@gmail.com)


Abstract
------
Default risk assessment for small companies is a tough problem in financial services. Recent efforts utilize advanced Heterogeneous Graph Neural Networks (HGNNs) with metapaths to exploit interactive features in corporate activities for risk analysis. However, few works are proposed for commercial banks. Given a real financial graph, how to detect corporate default risks? We identify two challenges for the task. (1) Massive noisy connections hinder HGNNs to achieve strong results. (2) Multiple semantic connections greatly increase transitive default risk, while existing aggregation schemes do not leverage such connection patterns. In this work, we propose a novel Heterogeneous Graph Co-Attention Network for corporate default risk assessment. Our model takes advantage of collaborative metapaths to distill risky features by a co-attentive aggregation mechanism. First, the local attention score models the importance of neighbors under each metapath by holistic metapath context. Second, the global attention score fuse local attention scores to filter valuable/noisy signals. Then, pairwise importance learning aims to enhance attention scores of multi-metapath neighbors for risky feature distillation. Extensive experiments on large-scale banking datasets demonstrate the effectiveness of our method.


Requirements
------

* python == 3.10.11
* pytorch == 2.0.1
* dgl == 1.1.0
* numpy == 1.24.1
* pandas == 2.1.1
* scikit-learn == 1.3.1
* scipy == 1.11.3


Usage
------

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

* Notice: Real-world financial datasets are from an anonymous bank under an NDA agreement, and thus we do not disclose more information of corporate users used in the papar. Here, the DBLP data are provided for running the model.
