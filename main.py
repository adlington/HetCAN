import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

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
        # model parameters
        "hidden": [128, 128],
        "activation": [torch.tanh, F.relu],
        "dropout": [0.5, 0.5, 0.5],
        "num_layers": [1, 1],
        "num_heads": 1,
        "residual": True,
        "batch_norm": True,
        "loss_weight": 0.5,
        "margin": 1,
        # sampling parameters
        "neighbor_sampling": 10,
        # "coef": 5,
        # training parameters
        "batch_size": 32,
        "opt_name": "Adam",
        "lr": 0.001,
        "weight_decay": 0.01,
        "eval_metric": "micro_f1",
        "epoch": 300,
        "early_stopping": 30,
        "device": "cuda"
    }
    # read the graph dataset
    filepath = os.path.join(os.path.dirname(__file__), "data/DBLP/")
    dataset = DBLPHeteroGraph(root=filepath)
    # edges for metapath context
    metapath_ctx = {"apa_i": [("author", "ap", "paper"), ("paper", "pa", "author")], 
                "apvpa_i": [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")],
                "aptpa_i": [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]}
    # edges for metapath instances
    metapath_inst = {"apa": [("author", "ap", "paper"), ("paper", "pa", "author")], 
                "apvpa": [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")],
                "aptpa": [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]}
    dataset.transform(metapath_ctx, metapath_inst)
    dataset.g = dgl.edge_type_subgraph(dataset.g, etypes=list(metapath_ctx.keys())+list(metapath_inst.keys()))
    # create the model
    metapath_info = {
        "apa": [["paper"], [("author", "ap", "paper"), ("paper", "pa", "author")]],
        "apvpa": [["paper", "venue", "paper"],  [("author", "ap", "paper"), ("paper", "pv", "venue"), ("venue", "vp", "paper"), ("paper", "pa", "author")]],
        "aptpa":[["paper", "term", "paper"], [("author", "ap", "paper"), ("paper", "pt", "term"), ("term", "tp", "paper"), ("paper", "pa", "author")]],
    }  # the node and edge types along metapaths
    metapath_dict = {"apa_i": "apa", "apvpa_i": "apvpa", "aptpa_i": "aptpa"}
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
    # device
    if params["device"]:
        device = torch.device(params["device"])
        model = model.to(device)
    else:
        device = None
    # loss function
    criterion = [nn.CrossEntropyLoss(), nn.MarginRankingLoss(margin=params["margin"])]
    # set the optimizer parameters
    opt = {"opt_name": params["opt_name"], "lr": params["lr"], "weight_decay": params["weight_decay"]}
    # set the local sampling parameters
    sampling_params = {"num_layers": params["num_layers"][0], "neighbor_sampling": params["neighbor_sampling"], "prob": None}
    # train the model
    trainer = HetCANTrainer(
        dataset = dataset,
        metapath_dict = metapath_dict,
        model = model,
        opt = opt,
        criterion = criterion,
        sampling_params = sampling_params,
        batch_size = [params["batch_size"], params["batch_size"]],
        device=device
    )
    state, result = trainer.train(epochs=params["epoch"], early_stopping=params["early_stopping"], eval_metric=params["eval_metric"], device=device)
    # save the model
    # torch.save(state, "hetcan.pth")