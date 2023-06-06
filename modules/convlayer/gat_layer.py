
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv

# Implementing GAT Convolution with Edge Feature Considered via DGL's GATConv Toolkit
class GATConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, econv=False,
                       num_heads=8, feat_drop=0., attn_drop=0.,):
        super(GATConvLayer, self).__init__()
        # considering edge feature       
        self.econv = econv
        if self.econv: 
            self.edge_apply_func = nn.Linear(input_dim, output_dim*num_heads)

        if dgl.__version__ < "0.5":
            self.gatconv = GATConv(input_dim, output_dim, num_heads, feat_drop, attn_drop)
        else:
            self.gatconv = GATConv(input_dim, output_dim, num_heads, feat_drop, attn_drop, allow_zero_in_degree=True)

    def forward(self, graph, nfeat, efeat=None):

        degs = (graph.in_degrees().float()+1).to(graph.device)

        # self loop features for graph convolution 
        sfeat = self.gatconv.fc(nfeat)/degs.view(-1, 1)
       
        nfeat = self.gatconv(graph, nfeat).flatten(1) + sfeat
      
        if self.econv and efeat is not None: # add edge feature
            graph.edata['efeat'] = self.edge_apply_func(efeat)
            graph.update_all(fn.copy_e('efeat', 'e'), fn.mean('e', 'h_e'))
            rst = nfeat + graph.ndata['h_e']
        else:
            rst = nfeat
        return rst

