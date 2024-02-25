import torch
import dgl
import os
import os.path as osp
import json
import pandas as pd
import numpy as np
from dgl.data.utils import save_graphs, load_graphs
from sklearn.model_selection import train_test_split
from dgl.transforms import BaseTransform, Compose
from dgl.transforms.module import update_graph_structure
import scipy.sparse as sp


def str_to_tensor(s):
    """
    Description
    -----------
    Convert features from str to tensor.

    Parameters
    ----------
    s: str, the input feature of str type.

    Returns
    -------
    torch.Tensor, feature tensor.
    """
    return torch.FloatTensor([float(i) for i in s.split(",")])

class DBLPHeteroGraph:
    """
    Description
    -----------
    Read the heterogeneous graph.

    Parameters
    ----------
    root: str, the data directory.
    """
    def __init__(self, root):
        self.root = root
        # self.raw_dir = osp.join(root, "raw")
        self.raw_dir = root
        self.processed_dir = osp.join(root, "processed")
        self.info()
        self.process()
        self.feat_info()

    def info(self):
        """
        Description
        -----------
        Get the graph info.
        """
        info = json.loads(open(osp.join(self.raw_dir, "info.dat")).read())
        self.node_info = info["node.dat"]["node type"]
        self.edge_info = info["link.dat"]["link type"]
        self.target = list(info["label.dat"]["node type"].keys())[0]
        meta = json.loads(open(osp.join(self.raw_dir, "meta.dat")).read())
        self.num_nodes_dict = {}
        for k, v in self.node_info.items():
            self.num_nodes_dict[v] = int(meta["Node Type_" + k])

    def process(self):
        """
        Description
        -----------
        Read graph data and get the heterogeneous graph.
        """
        if not osp.exists(self.processed_dir):
            os.mkdir(self.processed_dir)
        processed_file_graph = osp.join(self.processed_dir, "dblp_hetero.bin")
        processed_file_info = osp.join(self.processed_dir, "dblp_hetero.pth")
        # load data from saved file
        if osp.exists(processed_file_graph):
            self.load(processed_file_graph, processed_file_info)
        else:
            # get node features
            nodedf = pd.read_csv(osp.join(self.raw_dir, "node.dat"), names=["idx", "name", "ntype", "feat"], sep="\t")
            # re-index different types of nodes
            nodedf["new_idx"] = nodedf.groupby(["ntype"])["idx"].rank().astype("int") - 1
            # save node features of different types
            self.nfeat = {}
            ntypelist = nodedf[nodedf.feat.notnull()].ntype.unique().tolist()
            for ntype in ntypelist:
                node_tmp = nodedf[nodedf.ntype == ntype].feat.tolist()
                self.nfeat[self.node_info[str(ntype)]] = torch.cat([str_to_tensor(v).unsqueeze(0) for v in node_tmp], dim=0).to(torch.float32)

            # read edges
            edgedf = pd.read_csv(osp.join(self.raw_dir, "link.dat"), names=["src", "dst", "etype", "weight"], sep="\t")
            # save edge features of different types
            self.efeat = {}
            # reorganize the edge data in a dict
            data_dict = {}
            for etype, edict in self.edge_info.items():
                edge_tmp = edgedf[edgedf.etype == int(etype)][["src", "dst"]]
                src_tmp = nodedf[nodedf.ntype == int(edict["start"])][["idx", "new_idx"]].rename(columns={"idx":"src", "new_idx":"src_idx"})
                edge_tmp = pd.merge(edge_tmp, src_tmp, on=["src"], how="left")
                dst_tmp = nodedf[nodedf.ntype == int(edict["end"])][["idx", "new_idx"]].rename(columns={"idx":"dst", "new_idx":"dst_idx"})
                edge_tmp = pd.merge(edge_tmp, dst_tmp, on=["dst"], how="left")
                edge_tmp = edge_tmp.reset_index().rename(columns={"index": "eid"})
                s = edict["meaning"].split("-")
                e_tuple = (self.node_info[edict["start"]], s[0][0] + s[1][0], self.node_info[edict["end"]])
                data_dict[e_tuple] = (torch.LongTensor(edge_tmp.src_idx.values), torch.LongTensor(edge_tmp.dst_idx.values))
            
            # create the heterogeneous graph
            self.g = dgl.heterograph(data_dict, num_nodes_dict=self.num_nodes_dict)

            # train test split
            traindf = pd.read_csv(osp.join(self.raw_dir, "label.dat"), names=["idx", "name", "ntype", "label"], sep="\t")
            train_nodes, valid_nodes = train_test_split(traindf["idx"], test_size=0.2, random_state=42)
            self.valid_nodes = torch.from_numpy(valid_nodes.values).to(torch.long)
            self.train_nodes = torch.from_numpy(train_nodes.values).to(torch.long)
            testdf = pd.read_csv(osp.join(self.raw_dir, "label.dat.test"), names=["idx", "name", "ntype", "label"], sep="\t")
            self.test_nodes = torch.from_numpy(testdf.idx.values).to(torch.long)
            print("train set size:{}, valid set size:{}, test set size:{}".format(len(self.train_nodes), len(self.valid_nodes), len(self.test_nodes)))

            # read label
            df = pd.concat([traindf, testdf], axis=0).sort_values(by=["idx"], ascending=True)
            target_nodes_num = self.num_nodes_dict[self.node_info[self.target]]
            # fillna
            if df.shape[0] != target_nodes_num:
                df1 = pd.DataFrame(range(target_nodes_num), columns=["idx"])
                df = pd.merge(df1, df, on=["idx"], how="left")
                df["label"] = df["label"].fillna(int(df["label"].max())+1)
            self.label = torch.from_numpy(df.label.values).to(torch.long)

            # save graph
            self.save(processed_file_graph, processed_file_info)

    def feat_info(self):
        """
        Description
        -----------
        Get the initial feature vector size and the number of node/edge embeddings. 
        """
        self.tf_dict = {}
        self.emb_ndict = {}
        self.emb_edict = {}
        for ntype in self.g.ntypes:
            if ntype in self.nfeat.keys():
                self.tf_dict[ntype] = self.nfeat[ntype].shape[1]
            else:
                self.emb_ndict[ntype] = self.g.num_nodes(ntype=ntype)
        for etype in self.g.canonical_etypes:
            if etype in self.efeat.keys():
                self.tf_dict[etype] = self.efeat[etype].shape[1]
            else:
                self.emb_edict[etype] = 1

    def transform(self, metapath_ctx, metapath_inst, coef=5):
        """
        Description
        -----------
        Transform the graph and add the virtual edges between the metapath-based neighbors as well as along each metapath instance.

        Parameters
        ----------
        metapath_ctx: dict[str, list], the keys represent the names of virtual edges between the metapath-based neighbors and the values represent the canonical edge types of metapaths.
        metapath_inst: dict[str, list], the keys represent the names of virtual edges along each metapath instance and the values represent the canonical edge types of metapaths.
        coef: int, the coefficient for multi-metapath neighbors.
        """
        transformed_graph_file = osp.join(self.processed_dir, "dblp_mpgraph.bin")
        if osp.exists(transformed_graph_file):
            self.load(transformed_graph_file)
        else:
            transform = Compose([AddMetapathContextEdge(metapath_ctx, coef=coef), AddMetapathInstanceEdge(metapath_inst)])
            self.g = transform(self.g)
            self.save(transformed_graph_file)

    def save(self, processed_file_graph, processed_file_info=""):
        """
        Description
        -----------
        Save the graph and other info.

        Parameters
        ----------
        processed_file_graph: str, the saved graph file.
        processed_file_info: str, the saved info file.
        """
        if processed_file_info:
            info = {"train_idx": self.train_nodes,
                    "valid_idx": self.valid_nodes,
                    "test_idx": self.test_nodes,
                    "label": self.label,
                    "nfeat": self.nfeat,
                    "efeat": self.efeat
                    }
            torch.save(info, processed_file_info)
        save_graphs(processed_file_graph, self.g)
        
    def load(self, processed_file_graph, processed_file_info=""):
        """
        Description
        -----------
        Load the graph from saved file.

        Parameters
        ----------
        processed_file_graph: str, the saved graph file.
        processed_file_info: str, the saved info file.
        """
        self.g = load_graphs(processed_file_graph)[0][0]
        if processed_file_info:
            info = torch.load(processed_file_info)
            self.train_nodes = info["train_idx"]
            self.valid_nodes = info["valid_idx"]
            self.test_nodes = info["test_idx"]
            self.label = info["label"]
            self.nfeat = info["nfeat"]
            self.efeat = info["efeat"]

class AddMetapathContextEdge(BaseTransform):
    """
    Description
    -----------
    Add new edges for metapath context to an input graph based on given metapaths.

    Parameters
    ----------
    metapaths: dict[str, list], the metapaths to add, mapping a metapath name to a metapath.
    keep_orig_edges: bool, whether to keep the edges of the original graph.
    coef: int, coefficient for multi-metapath neighbors.
    attr_name: str, the edge attribute name for distinguishing the multi-metapath neighbors from the single-metapath neighbors.
    """
    def __init__(self, metapaths, keep_orig_edges=True, coef=5, attr_name="prob"):
        self.metapaths = metapaths
        self.keep_orig_edges = keep_orig_edges
        self.coef = coef
        self.attr_name = attr_name
    
    def __call__(self, g):
        """
        Description
        -----------
        Transform the original graph.

        Parameters
        ----------
        g: DGLGraph, the original graph.

        Returns
        -------
        new_g: DGLGraph, the transform graph.
        """
        data_dict = {}
        edata_dict = {}
        adj_dict = {}
        compose_adj = None

        # calculate multi-metapath adjacency matrix
        for meta_etype, metapath in self.metapaths.items():
            meta_g = dgl.metapath_reachable_graph(g, metapath)
            meta_adj = meta_g.adj_external(scipy_fmt="csr")
            adj_dict[meta_etype] = meta_adj
            if compose_adj is None:
                compose_adj = meta_adj
            else:
                compose_adj = compose_adj + meta_adj
        # multiply the coefficient and add the metapath neighbor graph
        for meta_etype, metapath in self.metapaths.items():
            tmp_adj = compose_adj.multiply(adj_dict[meta_etype])
            coef_adj = (tmp_adj>1).astype("int").multiply(tmp_adj)
            mp_adj = (tmp_adj + coef_adj*(self.coef-1)).tocoo()
            src_type = metapath[0][0]
            dst_type = metapath[-1][-1]
            data_dict[(src_type, meta_etype, dst_type)] = (torch.LongTensor(mp_adj.row), torch.LongTensor(mp_adj.col))
            edata_dict[meta_etype] = torch.FloatTensor(mp_adj.data)
        # keep the edges in the original graph
        if self.keep_orig_edges:
            for c_etype in g.canonical_etypes:
                data_dict[c_etype] = g.edges(etype=c_etype)
            new_g = update_graph_structure(g, data_dict, copy_edata=True)
        else:
            new_g = update_graph_structure(g, data_dict, copy_edata=False)
        # add the edge attribute
        for meta_etype, edata in edata_dict.items():
            new_g.edges[meta_etype].data[self.attr_name] = edata

        return new_g

class AddMetapathInstanceEdge(BaseTransform):
    """
    Description
    -----------
    Add all the metapath instances to an input graph based on given metapaths, where the intermediate node and edge ids are kept in the attributes("hn" and "he") of the new edges.
    (If the intermediate edge has no attributes, it will not be saved)

    Parameters
    ----------
    metapaths: dict[str, list], the metapaths to add, mapping an edge name to a metapath.
    keep_orig_edges: bool, whether to keep the edges of the original graph.
    """
    def __init__(self, metapaths, keep_orig_edges=True):
        self.metapaths = metapaths
        self.keep_orig_edges = keep_orig_edges

    def __call__(self, g):
        """
        Description
        -----------
        Transform the original graph.

        Parameters
        ----------
        g: DGLGraph, the original graph.

        Returns
        -------
        new_g: DGLGraph, the transform graph.
        """
        edge_dict = {}  # original edge data dict
        mp_data_dict = {}  # graph data dict
        mp_node_dict = {}  # node data dict
        mp_edge_dict = {}  # edge data dict
        self.mp_midtype_dict = {meta_etype: [[], []] for meta_etype in self.metapaths.keys()}  # the node and edge types along each metapath

        # convert all types of original edge data to dataframe
        for src_type, etype, dst_type in g.canonical_etypes:
            edge_array = np.concatenate([nids.unsqueeze(-1).numpy() for nids in g.edges(etype=etype)], axis=1)
            edge_df = pd.DataFrame(edge_array, columns=[src_type, dst_type])
            # if it has edge features, add the column eid
            if g.edges[(src_type, etype, dst_type)].data:
                edge_df = edge_df.reset_index().rename(columns={"index": "eid"})
                edge_df.columns = [src_type, "eid", dst_type]  # keep the order of columns
            edge_dict[etype] = edge_df

        # generate all the metapath instances by merging the dataframe
        for meta_etype, metapath in self.metapaths.items():
            for i in range(len(metapath)):
                if i == 0:
                    merge_df = edge_dict[metapath[i][1]].copy()
                    src_col = merge_df.columns[0]
                    dst_col = merge_df.columns[-1]
                    # rename the columns: for node columns, `{node_type_name}{number_in_metapath}{src_dst_ind}`; for edge columns, `eid{number_in_metapath}`
                    merge_df.rename(columns={src_col: src_col+str(i)+"0", dst_col: dst_col+str(i)+"1"}, inplace=True)
                    if "eid" in merge_df.columns:
                        self.mp_midtype_dict[meta_etype][1].append(metapath[i])
                        merge_df.rename(columns={"eid": "eid"+str(i)}, inplace=True)
                else:
                    tmp_df = edge_dict[metapath[i][1]].copy()
                    src_col = tmp_df.columns[0]
                    dst_col = tmp_df.columns[-1]
                    tmp_df.rename(columns={src_col: src_col+str(i)+"0", dst_col: dst_col+str(i)+"1"}, inplace=True)
                    if "eid" in tmp_df.columns:
                        self.mp_midtype_dict[meta_etype][1].append(metapath[i])
                        tmp_df.rename(columns={"eid": "eid"+str(i)}, inplace=True)
                    # merge on the destination node of last edge and the source node of next edge
                    old_col = merge_df.columns[-1]
                    new_col = tmp_df.columns[0]
                    merge_df = pd.merge(merge_df, tmp_df, left_on=old_col, right_on=new_col, how="inner")
                    self.mp_midtype_dict[meta_etype][0].append(metapath[i][0])
                    merge_df.drop(columns=new_col, inplace=True)
            
            # save all the metapath instance edges and their intermediate node and edge ids
            src_type = metapath[0][0]
            dst_type = metapath[-1][-1]
            mp_data_dict[(src_type, meta_etype, dst_type)] = (torch.LongTensor(merge_df.iloc[:, 0].values), torch.LongTensor(merge_df.iloc[:, -1].values))
            is_mid_edge = np.array([True if "eid" in col else False for col in merge_df.columns[1:-1]])
            mp_mid_nodes = merge_df.columns[1:-1][~is_mid_edge].tolist()
            mp_mid_edges = merge_df.columns[1:-1][is_mid_edge].tolist()
            if mp_mid_nodes:
                mp_node_dict[meta_etype] = torch.LongTensor(merge_df[mp_mid_nodes].values)
            if mp_mid_edges:
                mp_edge_dict[meta_etype] = torch.LongTensor(merge_df[mp_mid_edges].values)
            print("metapath: {}, path instances: {}, intermediate node types: {}, intermadiate edge types: {}".format(meta_etype, merge_df.shape[0], len(mp_mid_nodes), len(mp_mid_edges)))

        # keep the edges in the original graph
        if self.keep_orig_edges:
            for c_etype in g.canonical_etypes:
                mp_data_dict[c_etype] = g.edges(etype=c_etype)
            new_g = update_graph_structure(g, mp_data_dict, copy_edata=True)
        else:
            new_g = update_graph_structure(g, mp_data_dict, copy_edata=False)

        # save the intermediate node and edge ids which have original features
        new_g.edata["hn"] = mp_node_dict
        new_g.edata["he"] = mp_edge_dict
            
        return new_g
