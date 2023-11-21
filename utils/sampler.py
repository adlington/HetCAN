import torch
import dgl
from dgl.dataloading import BlockSampler
from dgl.transforms import to_block
from collections import defaultdict


def sample_instances(inst_eids, eids, topk):
    """
    Description
    -----------
    Sample metapath instances. 

    Parameters
    ----------
    inst_eids: tensor, edge ids for metapath instances.
    eids: tensor, edge ids for metapath-based neighbors of the given metapath instances.
    topk: int, sampling numbers.

    Returns
    -------
    sample_inst_eids: tensor, edge ids of sampling metapath instances.
    """
    # count the metapath instances for each metapath-based neighbor pairs
    count = torch.bincount(eids)
    all_eids = torch.arange(len(count), dtype=torch.long)
    # if # <= topk, collect all the instances
    ind1 = torch.isin(eids, all_eids[count<=topk])
    sample_inst_eids = inst_eids[ind1]
    # if # > topk, sample topk instances
    sample_eids = all_eids[count>topk]
    for e in sample_eids:
        inst_e = inst_eids[eids==e]
        random_ind = torch.randperm(len(inst_e))
        sample_inst_eids = torch.cat([sample_inst_eids, inst_e[random_ind[:topk]]], dim=0)
    return sample_inst_eids


class HetCANSampler(BlockSampler):
    """
    Description
    -----------
    Sampler that builds computational dependency of node representations via neighbor sampling for HetCAN.

    Parameters
    ----------
    metapath_dict: dict, the virtual edge names between metapath-based neighbors (key) and for metapath instances (value).
    fanouts: list[dict[etype, int]], list of neighbors to sample per edge type for each GNN layer, with the i-th element being the fanout for the i-th GNN layer.
    edge_dir: str, 
        can be either ``'in' `` where the neighbors will be sampled according to incoming edges, or ``'out'`` otherwise.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional to the edge feature value with the given name in ``g.edata``.  The feature must be a scalar on each edge.
        This argument is mutually exclusive with :attr:`mask`.  If you want to specify both a mask and a probability, consider multiplying the probability with the mask instead.
    mask : str, optional
        If given, a neighbor could be picked only if the edge mask with the given name in ``g.edata`` is True.  The data must be boolean on each edge.
        This argument is mutually exclusive with :attr:`prob`.  If you want to specify both a mask and a probability, consider multiplying the probability with the mask instead.
    replace : bool, default False
        Whether to sample with replacement
    prefetch_node_feats : list[str] or dict[ntype, list[str]], optional
        The source node data to prefetch for the first MFG, corresponding to the input node features necessary for the first GNN layer.
    prefetch_labels : list[str] or dict[ntype, list[str]], optional
        The destination node data to prefetch for the last MFG, corresponding to the node labels of the minibatch.
    prefetch_edge_feats : list[str] or dict[etype, list[str]], optional
        The edge data names to prefetch for all the MFGs, corresponding to the edge features necessary for all GNN layers.
    output_device : device, optional
        The device of the output subgraphs or MFGs.  Default is the same as the minibatch of seed nodes.
    """
    def __init__(
        self,
        metapath_dict,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super(HetCANSampler, self).__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.metapath_dict = metapath_dict
        self.metapaths = [v for k, v in metapath_dict.items()]
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        """
        Description
        -----------
        Sampling the MFGs for given graph and mini-batch target nodes.

        Parameters
        ----------
        g: DGLHeteroGraph, heterogeneous graph.
        seed_nodes: dict[ntype, tensor], the mini-batch target nodes.
        exclude_eids: dict[etype, tensor], edge ids to exclude.

        Returns
        -------
        seed_nodes: dict[ntype, tensor], sampling nodes in MFGs.
        output_nodes: dict[ntype, tensor], target nodes.
        blocks: list[MFG], MFGs.
        """
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            # sampling metapath-based neighbros
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            sample_edges = {}
            sample_inst_edges = defaultdict(list)
            for etype in self.metapath_dict.keys():
                meta_etype = frontier.to_canonical_etype(etype)
                sample_edges[etype] = frontier.edata[dgl.EID][meta_etype]
                edges = frontier.edges(etype=etype)
                # sampling metapath instances
                for inst_etype in self.metapaths:
                    inst_u, inst_v, inst_eids = g.edge_ids(u=edges[0], v=edges[1], return_uv=True, etype=inst_etype)
                    eids = frontier.edge_ids(u=inst_u, v=inst_v, return_uv=True, etype=etype)[-1]
                    assert len(eids) == len(inst_eids), f"metapath edge size {len(eids)}, metapath instance edge size {len(inst_eids)}"
                    sample_inst_eids = sample_instances(inst_eids, eids, fanout[meta_etype])
                    sample_inst_edges[inst_etype].append(sample_inst_eids)
            # drop duplicated sampling edges
            for inst_etype, edge_list in sample_inst_edges.items():
                sample_edges[inst_etype] = torch.unique(torch.cat(edge_list, 0))
            sg = g.edge_subgraph(sample_edges, relabel_nodes=False, store_ids=True, output_device=self.output_device)
            eid = sg.edata[dgl.EID]
            block = to_block(sg, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks