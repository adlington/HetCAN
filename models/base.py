import torch
import torch.nn as nn
import torch.nn.functional as F


class classifier(nn.Module):
    """
    Description
    -----------
    A MLP classifier.

    Parameters
    ----------
    in_dim: int, input layer size.
    hid_dim: int, hidden layer size.
    out_dim: int, output layer size.
    num_layers: int, number of layers. (Default: 2)
    activation: func or None, activation function, if None, no activation. (Default: F.relu)
    normalize: bool, whether the model applies batch normalization layers. (Default: True)
    dropout: float, the dropout rate in hidden layers. (Default: 0.0)
    """
    def __init__(self,
                in_dim,
                hid_dim,
                out_dim,
                num_layers = 2,
                activation = F.relu,
                normalize = True,
                dropout = 0.0):
        super(classifier, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.norm = normalize
        self.dropout = nn.Dropout(dropout)
        self.clf = nn.ModuleList()
        if num_layers > 1:
            self.clf.append(nn.Linear(in_dim, hid_dim))
            for i in range(num_layers-2):
                self.clf.append(nn.Linear(hid_dim, hid_dim))
            self.clf.append(nn.Linear(hid_dim, out_dim))
            if normalize:
                self.bn = nn.ModuleList([nn.BatchNorm1d(hid_dim) for i in range(num_layers-1)])
        else:
            self.clf.append(nn.Linear(in_dim, out_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for l0 in self.clf:
            l0.reset_parameters()
        if self.num_layers > 1 and self.norm:
            for l1 in self.bn:
                l1.reset_parameters()

    def forward(self, x):
        """
        Description
        -----------
        Forward propagation calculation for the model.

        Parameters
        ----------
        x: tensor, input features.

        Returns
        -------
        x: tensor, output results.
        """
        if self.num_layers > 1:
            if self.norm:
                for l, b in zip(self.clf[:-1], self.bn):
                    x = self.dropout(self.activation(b(l(x))))
            else:
                for l in self.clf[:-1]:
                    x = self.dropout(self.activation(l(x)))
        
        x = self.clf[-1](x)
        return x

class RSNEncoder(nn.Module):
    """
    Description
    -----------
    A metapath context encoder which uses RSNs.

    Parameters
    ----------
    in_dim: int, input size.
    out_dim: int, output size.
    rnn_type: str, the base RNN model name. (Default: "gru")
    """
    def __init__(self, in_dim, out_dim, rnn_type="gru"):
        super(RSNEncoder, self).__init__()
        if rnn_type == "rnn":
            self.rnn = nn.RNNCell(in_dim, out_dim)
        elif rnn_type == "gru":
            self.rnn = nn.GRUCell(in_dim, out_dim)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTMCell(in_dim, out_dim)
        self.w1 = nn.Linear(out_dim, out_dim, bias=False)
        self.w2 = nn.Linear(out_dim, out_dim, bias=False)

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        self.rnn.reset_parameters()
        self.w1.reset_parameters()
        self.w2.reset_parameters()

    def forward(self, x, hn=None):
        """
        Description
        -----------
        Forward propagation calculation for the encoder.

        Parameters
        ----------
        x: tensor, input feature sequence.
        hn: tensor or None, initial hidden state

        Returns
        -------
        hn: tensor, output results.
        """
        for i in range(len(x)):
            hn = self.rnn(x[i], hn)
            if i % 2 == 1:
                hn = self.w1(hn)+self.w2(x[i-1])
        return hn

class SemanticFusion(nn.Module):
    """
    Description
    -----------
    Semantic fusion layer.

    Parameters
    ----------
    in_dim: int, the input feature size.
    hid_dim: int, the metapath attention vector size. (Default: 128)
    batch: bool, whether average over the batch. (Default: False)
    """
    def __init__(self, in_dim, hid_dim=128, batch=False):
        super(SemanticFusion, self).__init__()
        self.batch = batch
        self.project = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, 1, bias=False)
        )

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for l in self.project:
            if isinstance(l, nn.Linear):
                l.reset_parameters()

    def forward(self, h):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        h: tensor, the input feature tensor.

        Returns
        -------
        out: tensor, the output feature tensor.
        """
        if self.batch:
            w = self.project(h).mean(0)                    # (num_metapath, 1)
            w = F.softmax(w, dim=0)                 # (num_metapath, 1)
            w = w.expand((h.shape[0],) + w.shape)  # (batch_size, num_metapath, 1)
        else:
            w = self.project(h)                    # (batch_size, num_metapath, 1)
            w = F.softmax(w, dim=1)                 # (batch_size, num_metapath, 1)
        out = (w * h).sum(1)                       # (batch_size, in_dim)
        
        return out