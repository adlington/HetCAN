import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import dgl
from dgl.dataloading import DataLoader
from .utils import Earlystopping
from .sampler import HetCANSampler


def get_optimizer(model, opt_name, lr, weight_decay):
    """
    Description
    -----------
    Define the optimizer.

    Parameters
    ----------
    model: model instance, the model for training.
    opt_name: str, optimizer name, ["SGD", "Adam", "AdamW"].
    lr: float, learning rate.
    weight_decay: float, weight_decay.

    Returns
    -------
    optimizer: optimizer instance, the optimizer.
    """
    if opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise RuntimeError("No optimizer named {} is available!".format(opt_name))

    return optimizer

def get_scheduler(optimizer, scheduler_name, step, **kwargs):
    """
    Description
    -----------
    Define the lr scheduler.

    Parameters
    ----------
    optimizer: optimizer instance, the optimizer.
    scheduler_name: str, the scheduler name, ["StepLR", "CosAnnLR", "CosAnnWR"].
    step: int, the period of different schedulers.

    Returns
    -------
    scheduler: lr_scheduler instance, the lr scheduler.
    """
    if scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=kwargs["gamma"])
    elif scheduler_name == "CosAnnLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step, eta_min=kwargs["min_lr"])
    elif scheduler_name == "CosAnnWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step, eta_min=kwargs["min_lr"])
    else:
        raise RuntimeError("No scheduler named {} is available!".format(scheduler_name))
    return scheduler


class HetCANTrainer:
    """
    Description
    -----------
    HetCAN model trainer.

    Parameters
    ----------
    dataset: dataset instance, the dataset of the input graph.
    metapath_dict: dict, the virtual edge names between metapath-based neighbors (key) and for metapath instances (value).
    model: model instance, the model for training.
    opt: dict[str, any], the parameters of the optimizer.
    criterion: nn.Module, loss function.
    sampling_params: dict[str, any], the parameters of neighbor sampler.
    batch_size: list[int], the batch size for training and evaluation. (Default: [128,128])
    lr_decay: dict[str, any] or None, the parameters of the lr_scheduler, if None, do not use lr decay. (Default: None)
    multi_nodes: bool, whether is the heterogeneous graph. (Default: True)
    device: device instance or None, if None, use cpu training. (Default: None)
    """
    def __init__(self, dataset, metapath_dict, model, opt, criterion, sampling_params, batch_size=[128, 128], lr_decay=None, device=None):
        self.dataset = dataset
        self.metapath_dict = metapath_dict
        self.model = model
        self.optimizer = get_optimizer(model=model, **opt)
        self.criterion = criterion
        self.batch_size = batch_size[0]
        self.eval_batch_size = batch_size[1]
        self.lr_decay = lr_decay
        if lr_decay is not None:
            self.scheduler = get_scheduler(self.optimizer, **lr_decay)
        self.sampling_params = sampling_params
        self.device = device

    def get_sampler(self):
        """
        Description
        -----------
        Get the sampler which can create the MFGs.

        Returns
        -------
        sampler: BlockSampler, the sampler instance.
        """
        # sample certain metapath_based neighbors for each layer
        sampling_dict = {}
        for meta_etype in self.dataset.g.canonical_etypes:
            if meta_etype[1] in self.metapath_dict.keys():
                sampling_dict[meta_etype] = self.sampling_params["neighbor_sampling"]
            else:
                sampling_dict[meta_etype] = 0
        sample_num = [sampling_dict]*self.sampling_params["num_layers"]
        sampler = HetCANSampler(metapath_dict=self.metapath_dict, fanouts=sample_num, prob=self.sampling_params["prob"])

        return sampler

    def get_dataloader(self, mode):
        """
        Description
        -----------
        Get the dataloader.

        Parameters
        ----------
        mode: str, the value should be in ["train", "train_eval", "valid", "test"], where "train" means training set for training, 
        "train_eval" means training set for evaluation, "valid" means validation set for evaluation and "test" means test set for evaluation.

        Returns
        -------
        data_loader: NodeDataLoader, a dataloader instance.
        """
        if mode == "train":
            nodes = self.dataset.train_nodes
        elif mode == "train_eval":
            nodes = self.dataset.train_nodes
        elif mode == "valid":
            nodes = self.dataset.valid_nodes
        elif mode == "test":
            nodes = self.dataset.test_nodes
        else:
            raise RuntimeError("No mode named {} is available!".format(mode))

        if self.device:
            nodes = nodes.to(self.device)
        sampler = self.get_sampler()
        nodes1 = {self.model.target:nodes}

        if mode == "train":
            data_loader = DataLoader(self.dataset.g, nodes1, sampler, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            data_loader = DataLoader(self.dataset.g, nodes1, sampler, batch_size=self.eval_batch_size)
        return data_loader

    def train(self, epochs=200, early_stopping=50, eval_metric="auc", device=None):
        """
        Description
        -----------
        Train the model with early stopping according to the performance on validation set.

        Parameters
        ----------
        epochs: int, the training epochs. (Default: 200)
        early_stopping: int, the early stopping rounds. (Default: 50)
        eval_metric: str, the evaluation metric which should be in ["auc", "ks", "micro_f1", "macro_f1"]. (Default: "auc")
        device: torch.device or None, if None, use cpu training. (Default: None)

        Returns
        -------
        state: dict[str, any], the information of the best model.
        result: dict[str, float], the evaluation results on the test set.
        """
        es = Earlystopping(early_stopping)
        best = 0
        start_epoch = 0
        train_loader = self.get_dataloader(mode="train")
        train_loader2 = self.get_dataloader(mode="train_eval")
        valid_loader = self.get_dataloader(mode="valid")
        test_loader = self.get_dataloader(mode="test")
        if device is not None:
            nfeat_dict = {node: feat.to(device) for node, feat in self.dataset.nfeat.items()}
            efeat_dict = {edge: feat.to(device) for edge, feat in self.dataset.efeat.items()}
        else:
            nfeat_dict = self.dataset.nfeat
            efeat_dict = self.dataset.efeat
        inputs_all = {self.model.target: nfeat_dict[self.model.target]}
        label = self.dataset.label

        es(best)
        print("model training start")
        start_time = time.time()
        for i in range(start_epoch+1, epochs+1):
            print("epoch:{} training start".format(i))
            t0 = time.time()
            self.model.fit(train_loader, inputs_all, label, self.optimizer, self.criterion, nfeat_dict=nfeat_dict, efeat_dict=efeat_dict, device=device)
            train_loss1, train_loss2, y_pred_train = self.model.predict(train_loader2, inputs_all, label, self.criterion, nfeat_dict=nfeat_dict, efeat_dict=efeat_dict, device=device)
            train_evaluator = self.evaluate(label[self.dataset.train_nodes], y_pred_train)
            valid_loss1, valid_loss2, y_pred_valid = self.model.predict(valid_loader, inputs_all, label, self.criterion, nfeat_dict=nfeat_dict, efeat_dict=efeat_dict, device=device)
            valid_evaluator = self.evaluate(label[self.dataset.valid_nodes], y_pred_valid)
            t1 = time.time()
            print("epoch:{} training end, train loss1:{:.4f}, train loss2:{:.4f}, train {}:{:.4f}, valid loss1:{:.4f}, valid loss2:{:.4f}, valid {}:{:.4f}, time used:{:.2f}s".format(i, train_loss1, train_loss2, eval_metric, train_evaluator[eval_metric], valid_loss1, valid_loss2, eval_metric, valid_evaluator[eval_metric], t1-t0))

            if valid_evaluator[eval_metric] > best:
                best = valid_evaluator[eval_metric]
                state = {"model": deepcopy(self.model.state_dict()),
                        "optimizer": deepcopy(self.optimizer.state_dict()),
                        "epoch": i,
                        "train_loss1": train_loss1,
                        "train_loss2": train_loss2,
                        "valid_loss1": valid_loss1,
                        "valid_loss2": valid_loss2,
                        }
                for metric, v in train_evaluator.items():
                    state["train_{}".format(metric)] = v
                for metric, v in valid_evaluator.items():
                    state["valid_{}".format(metric)] = v
                if self.lr_decay is not None:
                    state["scheduler"] = deepcopy(self.scheduler.state_dict())
            if self.lr_decay is not None:
                self.scheduler.step()
            es(valid_evaluator[eval_metric])
            if es.early_stop:
                break
        self.model.reset_parameters()
        self.model.load_state_dict(state["model"])
        test_loss1, test_loss2, y_pred_test = self.model.predict(test_loader, inputs_all, label, self.criterion, nfeat_dict=nfeat_dict, efeat_dict=efeat_dict, device=device)
        test_evaluator = self.evaluate(label[self.dataset.test_nodes], y_pred_test)
        end_time = time.time()
        # print("model:HetCAN, batch_size:{}, hidden size:{}, learning rate:{}, activation:{}, dropout:{}, weight decay:{}, normalization:{}, optimizer:{}, scheduler:{}, neighbor sampling:{}".format(params["batch_size"], params["hidden_size"], params["lr"], params["activation"].__name__, params["dropout"], params["weight_decay"], params["norm"], params["opt_name"], params["scheduler_name"], params["neighbor_sampling"]))
        print("training end, metric:{}, best valid:{:.4f}, train:{:.4f}, test:{:.4f}, time used:{:.2f}s".format(eval_metric, best, state["train_{}".format(eval_metric)], test_evaluator[eval_metric], end_time-start_time))

        result = {"y_pred": y_pred_test,
                "test_loss1": test_loss1,
                "test_loss2": test_loss2,
                }
        for metric, v in test_evaluator.items():
            result["test_{}".format(metric)] = v
        return state, result

    def evaluate(self, y_true, y_prob):
        """
        Description
        -----------
        Evaluation of the model prediction.

        Parameters
        ----------
        y_true: array-like of shape (n_samples,), the true label.
        y_prob: array-like of shape (n_samples,), the prediction probabilities for binary classification or the prediction labels for multi-class classification.

        Returns
        -------
        evaluator: dict[str, float], the evaluation results.
        """
        evaluator = {}
        if self.model.binary:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            evaluator["ks"] = max(abs(tpr-fpr))
            evaluator["auc"] = roc_auc_score(y_true, y_prob)
        else:
            evaluator["micro_f1"] = f1_score(y_true, y_prob, average="micro")
            evaluator["macro_f1"] = f1_score(y_true, y_prob, average="macro")

        return evaluator
