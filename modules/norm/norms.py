
import torch
import torch.nn as nn
from modules.norm.norm_node import *
from modules.norm.norm_graph import *
from modules.norm.supernorm import SuperNorm

class NormalizeGNN(nn.Module):
    def __init__(self, norm_type = 'batchnorm', embed_dim=300, affine=True):
        super(NormalizeGNN, self).__init__()

        self.norm_type = norm_type
        self.norm = None
        
        if norm_type == 'batchnorm':
            self.norm = BatchNorm(embed_dim, affine=affine)
        elif norm_type == 'supernorm':
            self.norm = SuperNorm(embed_dim, affine=affine)
        elif norm_type == 'graphnorm':
            self.norm = GraphNorm(embed_dim, affine=affine)
        elif norm_type == 'exprenorm':
            self.norm = ExpreNorm(embed_dim, affine=affine)
        elif 'groupnorm' in norm_type:
            self.norm = group_norm(dim_to_norm=embed_dim, num_groups=4, skip_weight=1e-2,model=norm_type)
        elif 'nodenorm' in norm_type:
            self.norm = node_norm(mode=norm_type)
        elif 'meannorm' in norm_type:
            self.norm = mean_norm(mode=norm_type)
        elif 'pairnorm' in norm_type:
            self.norm = pair_norm(mode=norm_type)

    def forward(self, graph, tensor):

        if self.norm_type == 'None':
            tensor = tensor
        elif self.norm_type in ['groupnorm', 'nodenorm', 'groupnorm-bn', 'nodenorm-bn']:
            tensor = self.norm(tensor)
        else: 
            tensor = self.norm(graph, tensor)

        norm_tensor = tensor
        return norm_tensor