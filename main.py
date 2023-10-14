import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="HetCAN Training")
#     # model parameters
#     parser.add_argument("--hidden", type=int, default=64, help="The embedding dimension")
#     parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio")
#     parser.add_argument("--l2_norm", action="store_false", default=True, help="No L2 normalization")
#     parser.add_argument("--batch_norm", action="store_false", default=True, help="No batch normalization")
#     # training parameters
#     parser.add_argument("--opt_name", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"], help="Optimizer to use")
#     parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
#     parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
#     parser.add_argument("--epoch", type=int, default=200, help="Training epochs")
#     parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
#     parser.add_argument("--eval_metric", type=str, default="micro_f1", choices=["micro_f1", "macro_f1", "auc", "ks"], help="Evaluation metric")
#     parser.add_argument("--cuda", action="store_true", default=False, help="GPU training")
#     args = parser.parse_args()
#     return args

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import dgl
    from HetCAN.utils.dataset import DBLPHeteroGraph
    from HetCAN.models.hetcan import HETCAN
    from HetCAN.utils.trainer import HetCANTrainer
    # set the parameters
    params = {
        # model
        "hidden": [128, 128],
        "activation": [torch.tanh, F.relu],
        "dropout": [0.5, 0.5, 0.5],
        "num_layers": [1, 1],
        "num_heads": 1,
        "residual": True,
        "batch_norm": True,
        "loss_weight": 0.5,
        "margin": 1,
        # sampling
        "neighbor_sampling": 10,
        "coef": 5,
        # training
        "batch_size": 32,
        "opt_name": "Adam",
        "lr": 0.001,
        "weight_decay": 0.01,
        "eval_metric": "micro_f1",
        "epoch": 300,
        "early_stopping": 30,
    }
    # read the graph dataset
    filepath = os.path.join(os.path.dirname(__file__), "data/DBLP/")
    metapath_info = {
        "apa": [["paper"], [("author", "ap", "paper"), ("paper", "pa", "author")]],
        "apvpa": [["paper", "venue", "paper"],  [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")]],
        "aptpa":[["paper", "term", "paper"], [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]],
    }
    metapath_dict = {"apa_i": "apa", "apvpa_i": "apvpa", "aptpa_i": "aptpa"}
    dataset = DBLPHeteroGraph(root=filepath)
    metapath_sis = {"apa_i": [("author", "ap", "paper"), ("paper", "pa", "author")], 
                "apvpa_i": [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")],
                "aptpa_i": [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]}
    metapath_mp = {"apa": [("author", "ap", "paper"), ("paper", "pa", "author")], 
                "apvpa": [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")],
                "aptpa": [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]}
    dataset.transform(metapath_sis, metapath_mp, coef=params["coef"])
    dataset.g = dgl.edge_type_subgraph(dataset.g, etypes=list(metapath_sis.keys())+list(metapath_mp.keys()))
    # create the model
    model = HETCAN(
            tf_dict=dataset.tf_dict, 
            emb_ndict=dataset.emb_ndict,
            emb_edict=dataset.emb_edict,
            hid_dim=params["hidden"], 
            out_dim=dataset.label.unique().shape[0], 
            num_layers=params["num_layers"],
            metapath_info=metapath_info,
            metapath_dict=metapath_dict,
            target=dataset.node_info[dataset.target],
            heads=[params["num_heads"]]*params["num_layers"][0],
            activation=params["activation"], 
            dropout=params["dropout"],
            residual=params["residual"],
            batch_norm=params["batch_norm"],
            loss_weight=params["loss_weight"]
        )
    # loss function
    criterion = [nn.CrossEntropyLoss(), nn.MarginRankingLoss(margin=params["margin"])]
    # set the optimizer parameters
    opt = {"opt_name": params["opt_name"], "lr": params["lr"], "weight_decay": params["weight_decay"]}
    # set the local sampling parameters
    sampling_params = {"num_layers": params["num_layers"][0], "neighbor_sampling": params["neighbor_sampling"], "prob": "prob"}
    # train the model
    trainer = HetCANTrainer(
        dataset = dataset,
        metapath_dict = metapath_dict,
        model = model,
        opt = opt,
        criterion = criterion,
        sampling_params = sampling_params,
        batch_size = [params["batch_size"], params["batch_size"]]
    )
    state, result = trainer.train(epochs=params["epoch"], early_stopping=params["early_stopping"], eval_metric=params["eval_metric"])
    # save the model
    # torch.save(state, "hetcan.pth")