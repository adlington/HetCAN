import torch
import dgl
from dgl.dataloading import BlockSampler
from dgl.transforms import to_block
from collections import defaultdict


def sample_instances(inst_eids, eids, topk):
    count = torch.bincount(eids)
    all_eids = torch.arange(len(count), dtype=torch.long)
    # <= topk
    ind1 = torch.isin(eids, all_eids[count<=topk])
    sample_inst_eids = inst_eids[ind1]
    # > topk
    sample_eids = all_eids[count>topk]
    for e in sample_eids:
        inst_e = inst_eids[eids==e]
        random_ind = torch.randperm(len(inst_e))
        sample_inst_eids = torch.cat([sample_inst_eids, inst_e[random_ind[:topk]]], dim=0)
    return sample_inst_eids

class SemanticCoSampler(BlockSampler):

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
        super().__init__(
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
        output_nodes = seed_nodes
        blocks = defaultdict(list)
        mp_seed_nodes = {mp: seed_nodes for mp in self.metapath_dict.keys()}
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                replace=self.replace,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            for etype in frontier.etypes:
                can_etype = frontier.to_canonical_etype(etype)
                if fanout[can_etype] == 0:
                    continue
                edges = frontier.edges(etype=etype)
                sample_edges = {meta_etype: g.edge_ids(u=edges[0], v=edges[1], return_uv=True, etype=meta_etype)[-1] for _, meta_etype in self.metapath_dict.items()}
                sample_edges[etype] = frontier.edata[dgl.EID][can_etype]
                sg_tmp = g.edge_subgraph(sample_edges, relabel_nodes=False, store_ids=True, output_device=self.output_device)
                etype_list = [etype]
                for _, meta_etype in self.metapath_dict.items():
                    if sg_tmp.num_edges(etype=meta_etype)>0:
                        etype_list.append(meta_etype)
                sg = sg_tmp.edge_type_subgraph(etypes=etype_list)
                eid = sg.edata[dgl.EID]
                block = to_block(sg, mp_seed_nodes[etype])
                block.edata[dgl.EID] = eid
                mp_seed_nodes[etype] = block.srcdata[dgl.NID]
                blocks[etype].insert(0, block)

        return mp_seed_nodes, output_nodes, blocks

    def assign_lazy_features(self, result):
        """Assign lazy features for prefetching."""
        input_nodes, output_nodes, blocks = result
        for etype in blocks.keys():
            blocks[etype] = super().assign_lazy_features((input_nodes, output_nodes, blocks[etype]))[2]
        return input_nodes, output_nodes, blocks


class CoMpSampler(BlockSampler):

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
        super().__init__(
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
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
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
                for inst_etype in self.metapaths:
                    inst_u, inst_v, inst_eids = g.edge_ids(u=edges[0], v=edges[1], return_uv=True, etype=inst_etype)
                    eids = frontier.edge_ids(u=inst_u, v=inst_v, return_uv=True, etype=etype)[-1]
                    assert len(eids) == len(inst_eids)
                    sample_inst_eids = sample_instances(inst_eids, eids, fanout[meta_etype])
                    sample_inst_edges[inst_etype].append(sample_inst_eids)
            for inst_etype, edge_list in sample_inst_edges.items():
                sample_edges[inst_etype] = torch.unique(torch.cat(edge_list, 0))
            sg = g.edge_subgraph(sample_edges, relabel_nodes=False, store_ids=True, output_device=self.output_device)
            eid = sg.edata[dgl.EID]
            block = to_block(sg, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks