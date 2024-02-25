import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
from .base import classifier, RSNEncoder, SemanticFusion


def groupby_mean(values, labels, max_len):
    """
    Description
    -----------
    MeanPooling for metapath instance embeddings.

    Parameters
    ----------
    values: tensor, the metapath instance embeddings.
    labels: tensor, the edge ids for each metapath instance embedding.
    max_len: int, the max number of metapath instances.

    Returns
    -------
    means: tensor, the metapath context embeddings.
    counts: tensor, the number of metapth instances for each pair of metapath neighbors.
    """
    label = labels.view((labels.shape[0],)+(1,)*(len(values.shape)-1))
    counts = torch.zeros((max_len,)+(1,)*(len(values.shape)-1), dtype=torch.float32).to(values.device)
    sums = torch.zeros((max_len,)+values.shape[1:], dtype=torch.float32).to(values.device)
    counts.scatter_add_(0, label, torch.ones_like(label, dtype=torch.float32).to(label.device))
    sums.scatter_add_(0, label.expand((-1,)+values.shape[1:]), values)
    div = torch.where(counts>0, counts, 1)
    means = sums/div
    return means, counts

class CoattnConv(nn.Module):
    """
    Description
    -----------
    The co-attentive aggregation layer.

    Parameters
    ----------
    tf_dict: dict[ntype/etype, int], initial node/edge feature sizes.
    emb_ndict: dict[ntype, int], the numbers of initial node embeddings without initial node features.
    emb_edict: dict[etype, int], the numbers of initial edge embeddings without initial edge features.
    in_dim: int, input size.
    out_dim: int, output size.
    target: str, target node type name.
    metapath_info: dict[etype, list[list[ntype], list[etype]]], the node and edge type names for each metapath.
    metapath_dict: dict, the virtual edge names between metapath-based neighbors (key) and for metapath instances (value).
    activation: list[func], activation funtions.
    feat_dropout: float, feature dropout rate.
    attn_dropout: float, attention dropout rate.
    residual: bool, whether to use residual connections.
    num_heads: int, number of heads for multi-head attention.
    is_first: bool, whether is the first layer.
    """
    def __init__(self,
                tf_dict, 
                emb_ndict,
                emb_edict,
                in_dim,
                out_dim,
                target,
                metapath_info,
                metapath_dict,
                activation = [torch.tanh, F.relu],
                feat_dropout = 0.0,
                attn_dropout = 0.0,
                residual = True,
                num_heads = 1,
                is_first = True
                ):
        super(CoattnConv, self).__init__()
        self.hetlinear = dglnn.HeteroLinear(tf_dict, out_dim*num_heads)
        self.nhetembs = dglnn.HeteroEmbedding(emb_ndict, out_dim*num_heads)
        self.ehetembs = dglnn.HeteroEmbedding(emb_edict, out_dim*num_heads)
        self.out_dim = out_dim
        self.target = target
        self.metapath_info = metapath_info
        self.metapath_dict = metapath_dict
        self.activation0 = activation[0]
        self.activation1 = activation[1]
        self.feat_dropout = nn.Dropout(feat_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual = residual
        self.num_heads = num_heads
        self.is_first = is_first
        self.attn = nn.ParameterDict()
        self.eta = nn.ParameterDict()
        for etype in metapath_info.keys():
            self.attn[etype] = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
            self.eta[etype] = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))
        self.delta = nn.Parameter(torch.FloatTensor(size=(1, num_heads, 1)))

        self.encoder = nn.ModuleDict()
        for etype in metapath_info.keys():
            self.encoder[etype] = RSNEncoder(out_dim*num_heads, out_dim*num_heads)
            
        self.fc_src = nn.ModuleDict()
        self.fc_edge = nn.ModuleDict()
        for etype in metapath_dict.keys():
            self.fc_src[etype] = nn.Linear(out_dim, out_dim)
            self.fc_edge[etype] = nn.Linear(out_dim, out_dim)

        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for k, l in self.hetlinear.linears.items():
            l.reset_parameters()
        self.nhetembs.reset_parameters()
        self.ehetembs.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        for etype, attn in self.attn.items():
            nn.init.xavier_normal_(attn, gain=gain)
        for etype, eta in self.eta.items():
            nn.init.xavier_normal_(eta, gain=gain)
        nn.init.xavier_normal_(self.delta, gain=gain)
        for etype, encoder in self.encoder.items():
            encoder.reset_parameters()
        for etype, layer in self.fc_src.items():
            layer.reset_parameters()
        for etype, layer in self.fc_edge.items():
            layer.reset_parameters()

    def merge_feat(self, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Merge the initial features and the embeddings.

        Parameters
        ----------
        nfeat_dict: dict[ntype, tensor], initial node features.
        efeat_dict: dict[etype, tensor], initial edge features.

        Returns
        -------
        nfeat_new: dict[ntype, tensor], node features.
        efeat_new: dict[etype, tensor], edge features.
        """
        nfeat_new = {}
        efeat_new = {}
        for k, v in nfeat_dict.items():
            nfeat_new[k] = v
        for k, v in efeat_dict.items():
            efeat_new[k] = v
        for ntype, emb in self.nhetembs.embeds.items():
            nfeat_new[ntype] = emb
        for etype, emb in self.ehetembs.embeds.items():
            raw_etype = self.ehetembs.raw_keys[etype]
            efeat_new[raw_etype] = emb
        return nfeat_new, efeat_new

    def forward(self, g, h, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph or List[MFG], the graph data.
        h: dict[ntype, tensor], the features of the target node type.
        nfeat_dict: dict[ntype, tensor], the initial node features.
        efeat_dict: dict[etype, tensor], the initial edge features.

        Returns
        -------
        out: tensor, the output tensor.
        mean_m: tensor, the mean values of attention scores for multi-metapath neighbors.
        mean_s: tensor, the mean values of attention scores for single-metapath neighbors.
        """
        nfeat_new, efeat_new = self.merge_feat(nfeat_dict, efeat_dict)
        with g.local_scope():
            if isinstance(h, tuple):
                inputs_src, inputs_dst = h[0], h[1]
            elif g.is_block:
                inputs_src, inputs_dst = h, h[:g.num_dst_nodes()]
            else:
                inputs_src = inputs_dst = h
            # feature transformation
            if self.is_first:
                inputs_src = self.hetlinear.linears[self.target](inputs_src)  
                inputs_dst = self.hetlinear.linears[self.target](inputs_dst)
            
            h_src = self.feat_dropout(inputs_src).view(-1, self.num_heads, self.out_dim)  # (num_src_node, num_heads, out_dim)
            h_dst = self.feat_dropout(inputs_dst).view(-1, self.num_heads, self.out_dim)  # (num_dst_node, num_heads, out_dim)
            g.srcdata.update({"es": h_src})
            g.dstdata.update({"ed": h_dst})
            # metapath instance encoding
            h_inst = {}
            for etype in self.metapath_info.keys():
                mid_nfeat_list, mid_efeat_list = self.get_midfeat_list(g, etype, nfeat_new, efeat_new)  # (num_edge, num_heads*out_dim)
                edges = g.edges(etype=etype)
                mid_nfeat_list = [inputs_src[edges[0]]]+mid_nfeat_list+[inputs_dst[edges[1]]]
                assert len(mid_nfeat_list) - len(mid_efeat_list) == 1
                h_inst[etype] = self.mp_encoder(mid_nfeat_list, mid_efeat_list, etype)  # (num_edge, num_heads, out_dim)
            # calculate the metapath-specific embeddings
            h_list = []
            mean_m_list = []
            mean_s_list = []
            for etype in self.metapath_dict.keys():
                elist = []
                meta_etype = g.to_canonical_etype(etype)
                # calculate local attention scores
                for inst_etype in self.metapath_info.keys():
                    h_mean, e = self.edge_score(g, h_inst[inst_etype], etype, inst_etype)  # (num_edge, num_heads, out_dim), (num_edge, num_heads, 1)
                    if inst_etype == self.metapath_dict[etype]:
                        g.edata["h"] = {meta_etype: h_mean}
                        elist.append((1-torch.sigmoid(self.delta))*self.activation0(e))
                    else:
                        elist.append(torch.sigmoid(self.delta)*self.activation0(e))
                # calculate global attention scores
                edge_score = torch.cat(elist, dim=-1).sum(dim=-1, keepdim=True)  # (num_edge, num_heads, 1)
                # compute attention weights
                sub_g = g.edge_type_subgraph(etypes=[meta_etype])
                g.edata["a"] = {meta_etype: self.attn_dropout(dglnn.functional.edge_softmax(sub_g, edge_score))}  # (num_edge, num_heads,1)
                # neighbor aggregation
                g.update_all(message_func=fn.u_mul_e("es", "a", "msg"), reduce_func=fn.sum("msg", "ft"), etype=etype)
                g.update_all(message_func=lambda x: {"msg2": x.data["h"]*x.data["a"]}, reduce_func=fn.sum("msg2", "ft2"),  etype=etype)  # (num_dst_node, num_heads, out_dim)
                rst = self.fc_src[etype](g.dstdata["ft"]) + self.fc_edge[etype](g.dstdata["ft2"])  # (num_dst_node, num_heads, out_dim)
                if self.residual:
                    rst = rst + h_dst  # (num_dst_node, num_heads, out_dim)
                if self.activation1:
                    rst = self.activation1(rst)
                rst = rst.view(rst.shape[0], -1)  # (num_dst_node, num_heads*out_dim)
                h_list.append(rst)
                # pairwise attention scores
                g.edata["score"] = {meta_etype: edge_score.mean(dim=1)}  # (num_edge, 1)
                g.update_all(message_func=lambda x: {"msg_m": (x.data["prob"]>1) * (x.data["score"].squeeze())}, reduce_func=fn.sum("msg_m", "m_sum"), etype=etype)
                g.update_all(message_func=lambda x: {"msg_mc": (x.data["prob"]>1).to(torch.float32)}, reduce_func=fn.sum("msg_mc", "m_cnt"), etype=etype)
                g.update_all(message_func=lambda x: {"msg_s": (x.data["prob"]==1) * (x.data["score"].squeeze())}, reduce_func=fn.sum("msg_s", "s_sum"), etype=etype)
                g.update_all(message_func=lambda x: {"msg_sc": (x.data["prob"]==1).to(torch.float32)}, reduce_func=fn.sum("msg_sc", "s_cnt"), etype=etype)
                mean_m = g.dstdata["m_sum"]/torch.where(g.dstdata["m_cnt"]>0, g.dstdata["m_cnt"], 1)  # (num_dst_node,)
                mean_s = g.dstdata["s_sum"]/torch.where(g.dstdata["s_cnt"]>0, g.dstdata["s_cnt"], 1)  # (num_dst_node,)
                # ignore results without both types
                ind = (g.dstdata["m_cnt"]>0)*(g.dstdata["s_cnt"]>0)
                mean_m_list.append(mean_m[ind])
                mean_s_list.append(mean_s[ind])
            # combine all metapath results
            out = torch.stack(h_list, dim=1)  # (batch_size, num_metapath, num_heads*out_dim)
            mean_m = torch.cat(mean_m_list, dim=0)  # (num_node,)
            mean_s = torch.cat(mean_s_list, dim=0)  # (num_node,)

            return out, mean_m, mean_s

    def get_midfeat_list(self, g, etype, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Generate the node embedding sequence and the edge embedding sequence for metapath instances.

        Parameters
        ----------
        g: DGLHeteroGraph or List[MFG], the graph data.
        etype: str, the edge type name for metapath instances.
        nfeat_dict: dict[ntype, tensor], the node features.
        efeat_dict: dict[etype, tensor], the edge features.

        Returns
        -------
        mid_nfeat_list: list[tensor], the node embedding sequence.
        mid_efeat_list: list[tensor], the edge embedding sequence.
        """
        # transform and concatenate the features of the specific node and edge types along the metapaths
        mid_nfeat_list = []
        mid_efeat_list = []
        i, j = 0, 0
        inst_etype = g.to_canonical_etype(etype)
        for mp_ntype in self.metapath_info[etype][0]:
            ind = g.edata["hn"][inst_etype]
            # if the mid node has original features
            if mp_ntype in self.hetlinear.linears.keys():
                mid_nfeat = self.hetlinear.linears[mp_ntype](nfeat_dict[mp_ntype])  # (num_node_all, num_heads, out_dim)
                mid_nfeat = mid_nfeat[ind[:, i]]                     
            else:
                mid_nfeat = nfeat_dict[mp_ntype](ind[:, i])  # (num_node_all, num_heads, out_dim)
            i = i + 1
            mid_nfeat_list.append(mid_nfeat)
        for mp_etype in self.metapath_info[etype][1]:
            # if the mid edge has original features
            if str(mp_etype) in self.hetlinear.linears.keys():
                if mp_etype == inst_etype:
                    eids = g.edata[dgl.EID][mp_etype]
                    mid_efeat = self.hetlinear.linears[str(mp_etype)](efeat_dict[mp_etype][eids])
                else:
                    ind = g.edata["he"][inst_etype]
                    mid_efeat = self.hetlinear.linears[str(mp_etype)](efeat_dict[mp_etype])  # (num_edge_all, num_heads, out_dim)                        
                    mid_efeat = mid_efeat[ind[:, j]]  # (num_edge, num_heads, out_dim)
                    j = j + 1
            else:
                # edge type embedding
                ind = torch.LongTensor([0]).to(device=efeat_dict[mp_etype].weight.device)
                mid_efeat = efeat_dict[mp_etype](ind)  # (1, num_heads, out_dim)
                mid_efeat = mid_efeat.repeat(g.num_edges(etype=etype), 1)  # (num_edge, num_heads, out_dim)
            mid_efeat_list.append(mid_efeat)

        return mid_nfeat_list, mid_efeat_list

    def mp_encoder(self, nfeat, efeat, etype):
        """
        Description
        -----------
        The metapath instance encoder.

        Parameters
        ----------
        nfeat: list[tensor], the node embedding sequence.
        efeat: list[tensor], the edge embedding sequence.
        etype: str, the edge type name.

        Returns
        -------
        out: tensor, metpath instance embeddings.
        """
        feat = []
        for i in range(len(nfeat)):
            feat.append(nfeat[i])
            if i < len(nfeat)-1:
                feat.append(efeat[i])
        out = torch.stack(feat, dim=0)  # (mp_len, num_edge, num_heads*out_dim)
        out = self.encoder[etype](out)  # (num_edge, num_heads*out_dim)
        out = out.view(-1, self.num_heads, self.out_dim)  # (num_edge, num_heads, out_dim)
        return out

    def edge_score(self, g, h, etype, inst_etype):
        """
        Description
        -----------
        Calculate the metapath context embeddings and the local attention scores.

        Parameters
        ----------
        g: DGLHeteroGraph or List[MFG], the graph data.
        h: tensor, the metapath instance embeddings.
        etype: str, the edge type name for metapath neighbors.
        inst_etype: str, the edge type name for metapath instances.

        Returns
        -------
        h_mean: tensor, the metapath context embeddings.
        e: tensor, the edge embedding sequence.
        """
        # calculate the metapath context embeddings
        if g.device == torch.device("cpu"):
            u0, v0 = g.edges(etype=inst_etype)
            u1, v1, eids = g.edge_ids(u=u0, v=v0, return_uv=True, etype=etype)
            inst_eids = g.edge_ids(u=u1, v=v1, return_uv=True, etype=inst_etype)[-1]
            has_edge = torch.isin(torch.arange(u0.shape[0], dtype=torch.long), inst_eids)
        else:
            g1 = g.cpu()
            u0, v0 = g1.edges(etype=inst_etype)
            u1, v1, eids = g1.edge_ids(u=u0, v=v0, return_uv=True, etype=etype)
            inst_eids = g1.edge_ids(u=u1, v=v1, return_uv=True, etype=inst_etype)[-1]
            has_edge = torch.isin(torch.arange(u0.shape[0], dtype=torch.long), inst_eids)
            eids = eids.to(g.device)
            has_edge = has_edge.to(g.device)
        assert len(eids) == len(h[has_edge])
        h_mean, counts = groupby_mean(h[has_edge], eids, g.num_edges(etype=etype))  # (num_edge, num_heads, out_dim),  (num_edge, 1, 1)
        mp_ind = counts>0  # (num_edge, 1, 1)
        h_mean = self.feat_dropout(h_mean)  # (num_edge, num_heads, out_dim)
        # calculate the local attention scores
        u, v = g.edges(etype=etype)
        e1 = (g.srcdata["es"][u]*g.dstdata["ed"][v]).sum(dim=-1, keepdim=True)  # (num_edge, num_heads, 1)
        e1 = torch.sigmoid(self.eta[inst_etype])*e1  # (num_edge, num_heads, 1)
        e2 = (self.attn[inst_etype]*h_mean).sum(dim=-1, keepdim=True)  # (num_edge, num_heads, 1)
        e2 = (1 - torch.sigmoid(self.eta[inst_etype]))*e2  # (num_edge, num_heads, 1)
        e = mp_ind*(e1+e2)  # (num_edge, num_heads, 1)
        return h_mean, e

class HETCANLayer(nn.Module):
    """
    Description
    -----------
    The HetCAN layer.

    Parameters
    ----------
    tf_dict: dict[ntype/etype, int], initial node/edge feature sizes.
    emb_ndict: dict[ntype, int], the numbers of initial node embeddings without initial node features.
    emb_edict: dict[etype, int], the numbers of initial edge embeddings without initial edge features.
    hid_dim: int, hidden size.
    out_dim: int, output size.
    metapath_info: dict[etype, list[list[ntype], list[etype]]], the node and edge type names for each metapath.
    metapath_dict: dict, the virtual edge names between metapath-based neighbors (key) and for metapath instances (value).
    target: str, target node type name.
    activation: list[func], activation funtions.
    feat_dropout: float, feature dropout rate.
    attn_dropout: float, attention dropout rate.
    num_heads: int, number of heads for multi-head attention.
    residual: bool, whether to use residual connections.
    is_first: bool, whether is the first layer.
    is_last: bool, whether is the last layer.
    """
    def __init__(self,
                tf_dict, 
                emb_ndict,
                emb_edict,
                hid_dim, 
                out_dim,
                metapath_info,
                metapath_dict,
                target,
                activation = [torch.tanh, F.relu], 
                feat_dropout = 0.0,
                attn_dropout = 0.0,
                num_heads = 1,
                residual = True,
                is_first = True,
                is_last = False
                ):
        super(HETCANLayer, self).__init__()
        # co-attentive aggregation
        self.conv = CoattnConv(tf_dict, emb_ndict, emb_edict, hid_dim, hid_dim, target, metapath_info, metapath_dict,
                activation, feat_dropout, attn_dropout, residual, num_heads, is_first)
        # semanticc fusion
        self.sem_fusion = SemanticFusion(hid_dim*num_heads, hid_dim, batch=False)
        if is_last:
            self.fc = nn.Linear(hid_dim*num_heads, out_dim)
        else:
            self.fc = nn.Linear(hid_dim*num_heads, out_dim*num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        self.conv.reset_parameters()
        self.sem_fusion.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, g, h, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph or List[MFG], the graph data.
        h: dict[ntype, tensor], the features of the target node type.
        nfeat_dict: dict[ntype, tensor], the initial node features.
        efeat_dict: dict[etype, tensor], the initial edge features.

        Returns
        -------
        out: tensor, the output tensor.
        mean_m: tensor, the mean values of attention scores for multi-metapath neighbors.
        mean_s: tensor, the mean values of attention scores for single-metapath neighbors.
        """
        out, mean_m, mean_s = self.conv(g, h, nfeat_dict, efeat_dict)
        out = self.sem_fusion(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out, mean_m, mean_s

class HETCAN(nn.Module):
    """
    Description
    -----------
    The HetCAN model.

    Parameters
    ----------
    tf_dict: dict[ntype/etype, int], initial node/edge feature sizes.
    emb_ndict: dict[ntype, int], the numbers of initial node embeddings without initial node features.
    emb_edict: dict[etype, int], the numbers of initial edge embeddings without initial edge features.
    hid_dim: list[int], hidden size.
    out_dim: int, output size.
    num_layers: list[int], the number of layers.
    metapath_info: dict[etype, list[list[ntype], list[etype]]], the node and edge type names for each metapath.
    metapath_dict: dict, the virtual edge names between metapath-based neighbors (key) and for metapath instances (value).
    target: str, target node type name.
    heads: list[int], number of heads for multi-head attention.
    activation: list[func], activation funtions.
    dropout: list[float], dropout rate.
    residual: bool, whether to use residual connections.
    batch_norm: bool, whether uses the batch normalization in MLP.
    loss_weight: float, the weight of margin loss.
    """
    def __init__(self,
                tf_dict, 
                emb_ndict,
                emb_edict,
                hid_dim, 
                out_dim, 
                num_layers,
                metapath_info,
                metapath_dict,
                target,
                heads,
                activation = [torch.tanh, F.relu], 
                dropout = [0.0, 0.0, 0.0],
                residual = True,
                batch_norm = True,
                loss_weight = 0.5
                ):
        super(HETCAN, self).__init__()
        self.target = target
        self.binary = out_dim < 3
        self.convs = nn.ModuleList()
        if num_layers[0] == 1:
            self.convs.append(HETCANLayer(tf_dict, emb_ndict, emb_edict, hid_dim[0], hid_dim[0], metapath_info, metapath_dict, target,
                activation, dropout[0], dropout[1], heads[0], residual, True, True))
        else:
            self.convs.append(HETCANLayer(tf_dict, emb_ndict, emb_edict, hid_dim[0], hid_dim[0], metapath_info, metapath_dict, target,
                activation, dropout[0], dropout[1], heads[0], residual, True, False))
            for i in range(num_layers[0]-1):
                if i < num_layers[0]-2:
                    self.convs.append(HETCANLayer(tf_dict, emb_ndict, emb_edict, hid_dim[0], hid_dim[0], metapath_info, metapath_dict, target,
                activation, dropout[0], dropout[1], heads[i+1], residual, False, False))
                else:
                    self.convs.append(HETCANLayer(tf_dict, emb_ndict, emb_edict, hid_dim[0], hid_dim[0], metapath_info, metapath_dict, target,
                activation, dropout[0], dropout[1], heads[i+1], residual, False, True))
        self.clf = classifier(in_dim=hid_dim[0], hid_dim=hid_dim[1], out_dim=out_dim, num_layers=num_layers[1], activation=activation[1], normalize=batch_norm, dropout=dropout[2])
        self.loss_weight = loss_weight
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.
        """
        for conv in self.convs:
            conv.reset_parameters()
        self.clf.reset_parameters()

    def forward(self, g, h, nfeat_dict, efeat_dict):
        """
        Description
        -----------
        Forward computation.

        Parameters
        ----------
        g: DGLHeteroGraph or List[MFG], the graph data.
        h: dict[ntype, tensor], the features of the target node type.
        nfeat_dict: dict[ntype, tensor], the initial node features.
        efeat_dict: dict[etype, tensor], the initial edge features.

        Returns
        -------
        out: tensor, the output tensor.
        mean_m: tensor, the mean values of attention scores for multi-metapath neighbors.
        mean_s: tensor, the mean values of attention scores for single-metapath neighbors.
        """
        mean_m_list = []
        mean_s_list = []
        for conv, block in zip(self.convs, g):
            h, mean_m, mean_s = conv(block, h, nfeat_dict, efeat_dict)
            if mean_m.shape[0]>0:
                mean_m_list.append(mean_m)
                mean_s_list.append(mean_s)

        mean_m = torch.cat(mean_m_list, dim=0)
        mean_s = torch.cat(mean_s_list, dim=0)
        out = self.clf(h)
        
        return out, mean_m, mean_s

    def fit(self, dataloader, inputs_all, label, optimizer, criterion, device=None, log=0, **kwargs):
        """
        Description
        -----------
        Train the HetCAN model with given dataloader.

        Parameters
        ----------
        dataloader: dataLoader instance, dataloader for batch-iterating over a set of training nodes, generating the list of message flow graphs (MFGs) as computation dependency of the said minibatch.
        inputs_all: dict[str, tensor], the node features of all target type in the graph.
        label: tensor, the labels of all target type of nodes in the graph.
        optimizer: optimizer, the optimizer for training.
        criterion: list[nn.Module], the loss function for training.
        device: device instance or None, if None, use cpu training. (Default: None)
        log: int, the number of batches to print the training log, if set to zero, no log prints. (Default: 0)
        **kwargs: other forward parameters of the model.
        """
        self.train()
        total_loss = 0
        datasize = len(dataloader.dataset)
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            if isinstance(input_nodes, dict):
                input_nodes = input_nodes[self.target]
            # inputs = {self.target: inputs_all[self.target][input_nodes]}
            inputs = inputs_all[self.target][input_nodes]
            if isinstance(output_nodes, dict):
                output_nodes = output_nodes[self.target]
            y = label[output_nodes]
            if device is not None:
                # inputs = {k: v.to(device) for k, v in inputs.items()}
                inputs = inputs.to(device)
                y = y.to(device)
                blocks = [block.to(device) for block in blocks]
            logits, mean_m, mean_s = self.forward(blocks, inputs, **kwargs)
            loss1 = criterion[0](logits, y)
            loss2 = criterion[1](mean_m, mean_s, torch.ones_like(mean_m))
            loss = loss1+self.loss_weight*loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate loss and accuracy if log is required
            if log > 0:
                total_loss += loss.item() * len(output_nodes)
                acc = (logits.argmax(1) == y).float().mean()
                train_size = len(output_nodes)*(step+1) if len(output_nodes)*(step+1) < datasize else datasize

                if (step+1) % log == 0:
                    print("step:{},data:{}/{},loss:{:.4f},acc:{:.4f}".format(step+1,train_size,datasize,total_loss,acc))

    def predict(self, dataloader, inputs_all, label, criterion, device=None, **kwargs):
        """
        Description
        -----------
        Predict the HetCAN model with given dataloader.

        Parameters
        ----------
        dataloader: dataLoader instance, dataloader for batch-iterating over a set of training nodes, generating the list of message flow graphs (MFGs) as computation dependency of the said minibatch.
        inputs_all: dict[str, tensor], the node features of all target type in the graph.
        label: tensor, the labels of all target type of nodes in the graph.
        criterion: list[nn.Module], the loss function.
        device: device instance or None, if None, use cpu for inference. (Default: None)
        **kwargs: other forward parameters of the model.

        Returns
        -------
        total_loss1: float, average classification loss.
        total_loss2: float, average margin loss.
        y_prob: List[float] or List[int], positive probability or predicted labels of target nodes.
        """
        self.eval()
        total_loss1 = 0
        total_loss2 = 0
        y_pred_list = []
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in dataloader:
                if isinstance(input_nodes, dict):
                    input_nodes = input_nodes[self.target]
                # inputs = {self.target: inputs_all[self.target][input_nodes]}
                inputs = inputs_all[self.target][input_nodes]
                if isinstance(output_nodes, dict):
                    output_nodes = output_nodes[self.target]
                y = label[output_nodes]
                if device is not None:
                    # inputs = {k: v.to(device) for k, v in inputs.items()}
                    inputs = inputs.to(device)
                    y = y.to(device)
                    blocks = [block.to(device) for block in blocks]
                logits, mean_m, mean_s = self.forward(blocks, inputs, **kwargs)
                loss1 = criterion[0](logits, y)
                loss2 = criterion[1](mean_m, mean_s, torch.ones_like(mean_m))
                total_loss1 += loss1.item() * len(output_nodes)
                total_loss2 += loss2.item() * mean_m.shape[0]
                if self.binary:
                    y_pred_list.append(F.softmax(logits, dim=1)[:, 1].cpu())
                else:
                    y_pred_list.append(logits.argmax(1).cpu())

            total_loss1 /= len(dataloader.dataset)
            total_loss2 /= len(dataloader.dataset)
            y_prob = torch.cat(y_pred_list, dim=0)

            return total_loss1, total_loss2, y_prob