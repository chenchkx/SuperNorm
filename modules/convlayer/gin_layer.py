
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv
from modules.norm.norms import NormalizeGNN

# Implementing GIN Convolution with Edge Feature Considered via DGL's GINConv Toolkit
class GINConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, econv=False, mlp_apply=True, aggregator_type='sum'):
        super(GINConvLayer, self).__init__()
        # considering edge feature
        self.econv = econv
        if self.econv: 
            self.edge_apply_func = nn.Linear(input_dim, output_dim)
        if mlp_apply:
            apply_func = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),
                                    nn.ReLU(),
                                    nn.Linear(input_dim, output_dim)
                                    )
        else:
            apply_func =  nn.Linear(input_dim, output_dim)

        self.ginconv = GINConv(apply_func=apply_func, aggregator_type=aggregator_type)


    def forward(self, graph, nfeat, efeat=None):

        ginconv_nfeat = self.ginconv(graph, nfeat)

        if self.econv and efeat is not None: # add edge feature
            graph.edata['efeat'] = self.edge_apply_func(efeat)
            graph.update_all(fn.copy_e('efeat', 'e'), fn.sum('e', 'h_e'))
            rst = ginconv_nfeat + graph.ndata['h_e']
        else:
            rst = ginconv_nfeat    

        return rst
