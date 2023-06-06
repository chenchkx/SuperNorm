

import torch
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class NodeEncoder(nn.Module):
    def __init__(self, dataset_name, embed_dim):
        super(NodeEncoder, self).__init__()

        self.embed_dim = embed_dim
        if 'zinc' == dataset_name:
            self.node_encoder = nn.Embedding(28, embed_dim)
        elif 'ppa' in dataset_name:
            self.node_encoder = nn.Linear(7, embed_dim)
        elif 'ogbg-mol' in dataset_name:
            self.node_encoder = AtomEncoder(embed_dim)
        elif 'imdb-b' in dataset_name:
            self.node_encoder = nn.Linear(128, embed_dim)
            
    def forward(self, tensor):
        if tensor is None:
            return tensor
        else:
            return self.node_encoder(tensor)


class EdgeEncoder(nn.Module):
    def __init__(self, dataset_name, embed_dim=300):
        super(EdgeEncoder, self).__init__()

        self.embed_dim = embed_dim
        if 'zinc' == dataset_name:
            self.edge_encoder = nn.Embedding(4, embed_dim)
        elif 'ogbg-mol' in dataset_name:
            self.edge_encoder = BondEncoder(embed_dim)
        elif 'ppa' in dataset_name:
            self.edge_encoder = nn.Linear(7, embed_dim)
        elif 'imdb-b' in dataset_name:
            self.node_encoder = nn.Linear(128, embed_dim)
        # for node level prediction 
        elif 'cora' in dataset_name:
            self.edge_encoder = nn.Linear(1433, embed_dim)
        elif 'citeseer' in dataset_name:
            self.edge_encoder = nn.Linear(3703, embed_dim)
        elif 'pubmed' in dataset_name:
            self.edge_encoder = nn.Linear(500, embed_dim)
        elif 'ogbn-proteins' in dataset_name:
            self.edge_encoder = nn.Linear(8, embed_dim)
        elif 'ogbn-arxiv' in dataset_name:
            self.edge_encoder = nn.Linear(128, embed_dim)

    def forward(self, tensor):
        if tensor is None:
            return tensor
        else:
            return self.edge_encoder(tensor)
