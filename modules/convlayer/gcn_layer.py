
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GraphConv

# Implementing GCN Convolution with Edge Feature Considered via DGL's GCNConv Toolkit
class GCNConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, econv=False):
        super(GCNConvLayer, self).__init__()
        # edge feature convolution
        self.econv = econv
        if self.econv: 
            self.edge_apply_func = nn.Linear(input_dim, output_dim)
        
        if dgl.__version__ < "0.5":
            self.gcnconv = GraphConv(input_dim, output_dim)
        else:
            self.gcnconv = GraphConv(input_dim, output_dim, allow_zero_in_degree=True)

    def forward(self, graph, nfeat, efeat=None, edge_weight=None):
        degs = (graph.in_degrees().float()+1).to(graph.device)

        # self loop features for graph convolution 
        sfeat = (torch.matmul(nfeat, self.gcnconv.weight)+self.gcnconv.bias)/degs.view(-1, 1)
        
        nfeat = self.gcnconv(graph, nfeat, edge_weight=edge_weight) + sfeat

        if self.econv: # add edge feature
            graph.edata['efeat'] = self.edge_apply_func(efeat)
            graph.update_all(fn.copy_e('efeat', 'e'), fn.mean('e', 'h_e'))
            rst = nfeat + graph.ndata['h_e']
        else:
            rst = nfeat
        return rst


# Implementing GCN Convolution with Edge Feature Considered via DGL's GCNConv Toolkit
class GCNLinkConv(nn.Module):
    def __init__(self, input_dim, output_dim, econv=False):
        super(GCNLinkConv, self).__init__()
        # edge feature convolution
        self.econv = econv
        if self.econv: 
            self.edge_apply_func = nn.Linear(input_dim, output_dim)
        

        self.gcnconv = GraphConv(input_dim, output_dim, norm='none', allow_zero_in_degree=True)

    def forward(self, graph, nfeat, efeat, edge_weight=None):
        degs = (graph.in_degrees().float()+1).to(graph.device)

        # self loop features for graph convolution 
        sfeat = (torch.matmul(nfeat, self.gcnconv.weight)+self.gcnconv.bias)/degs.view(-1, 1)
        
        nfeat = self.gcnconv(graph, nfeat, edge_weight=edge_weight) + sfeat

        if self.econv and efeat is not None: # add edge feature
            graph.edata['efeat'] = self.edge_apply_func(efeat)
            graph.update_all(fn.copy_e('efeat', 'e'), fn.mean('e', 'h_e'))
            rst = nfeat + graph.ndata['h_e']
        else:
            rst = nfeat
        return rst



# GCNConv with edge feature embedding
class GCNConvLayer_WithEFeat(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GCNConvLayer_WithEFeat, self).__init__()
        
        self.project_edge_feat = nn.Linear(edge_dim, node_dim)
        self.project_node_feat = nn.Linear(node_dim, node_dim)
        self.project_residual =  nn.Embedding(1, node_dim)


    def forward(self, graph, nfeat, efeat):

        graph = graph.local_var()
        degs = (graph.in_degrees().float()+1).to(graph.device)
        graph.ndata['norm'] = torch.pow(degs, -0.5).unsqueeze(-1) 
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))
        norm = graph.edata.pop('norm')

        nfeat = self.project_node_feat(nfeat)
        efeat = self.project_edge_feat(efeat)
        graph.ndata['feat'] = nfeat
        graph.apply_edges(fn.copy_u('feat', 'e'))
        graph.edata['e'] = norm * (graph.edata['e'] + efeat)
        graph.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'feat'))

        residual_nfeat = nfeat + self.project_residual.weight
        residual_nfeat = residual_nfeat * 1. / degs.view(-1, 1)

        rst = graph.ndata['feat'] + residual_nfeat
        return rst