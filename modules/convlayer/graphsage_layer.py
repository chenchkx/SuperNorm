
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import SAGEConv

# Implementing GraphSage Convolution with Edge Feature Considered via DGL's SAGEConv Toolkit
class GraphSageLayer(nn.Module):
    def __init__(self, input_dim, output_dim, econv=False):
        super(GraphSageLayer, self).__init__()
        # edge feature convolution
        self.econv = econv
        if self.econv: 
            self.edge_apply_func = nn.Linear(input_dim, output_dim)
        
        self.sageconv = SAGEConv(input_dim, output_dim, aggregator_type='gcn')


    def forward(self, graph, nfeat, efeat=None, edge_weight=None):
        
        degs = (graph.in_degrees().float()+1).to(graph.device)

        # self loop features for graph convolution 
        sfeat = self.sageconv.fc_neigh(nfeat)/degs.view(-1, 1)

        nfeat = self.sageconv(graph, nfeat, edge_weight=edge_weight) + sfeat

        if self.econv and efeat is not None: # add edge feature
            graph.edata['efeat'] = self.edge_apply_func(efeat)
            graph.update_all(fn.copy_e('efeat', 'e'), fn.mean('e', 'h_e'))
            rst = nfeat + graph.ndata['h_e']
        else:
            rst = nfeat
        return rst

