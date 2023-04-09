

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops import segment

class GraphNorm(nn.Module):
    ### Graph norm implemented by dgl toolkit
    ### more details please refer to the ICML2021 paper: Graphnorm: A principled approach to accelerating graph neural network training
    ### Source code avaliable at https://github.com/lsj2408/GraphNorm
    def __init__(self, embed_dim=300, affine=True):
        super(GraphNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.mean_scale = nn.Parameter(torch.zeros(embed_dim))
        self.affine = affine
        
    def forward(self, graph, tensor):

        batch_list = graph.batch_num_nodes()
        batch_size = len(batch_list)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        # sub = tensor - mean
        sub = tensor - mean * self.mean_scale     
        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class ExpreNorm(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ExpreNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else: 
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, graph, tensor):  

        tensor = tensor*graph.ndata['square_n'].unsqueeze(1)

        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        bn_training = True if self.training else ((self.running_mean is None) and (self.running_var is None))
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None: 
                self.num_batches_tracked = self.num_batches_tracked + 1  
                if self.momentum is None:  
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self.momentum
        results = F.batch_norm(
                    tensor, self.running_mean, self.running_var, None, None,
                    bn_training, exponential_average_factor, self.eps)

        if self.affine:
            results = self.weight*results + self.bias
        else:
            results = results
        return results


class BatchNorm(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(BatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        # Here affine is False, just centering and scaling operations
        self.centerScale = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, graph, tensor):  

        cens_tensor = self.centerScale(tensor)

        if self.affine:
            rst = self.weight*cens_tensor+self.bias
        else:
            rst = cens_tensor
        return rst    


class InstanceNorm(nn.Module):

    def __init__(self, embed_dim=300, affine=True):
        super(InstanceNorm, self).__init__()

        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.affine = affine

    def repeat(self, tensor, batch_list):
        batch_size = len(batch_list)
        batch_indx = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        return tensor[batch_indx]

    def forward(self, graph, tensor):
        batch_list = graph.batch_num_nodes()      
        mean = segment.segment_reduce(batch_list, tensor, reducer='mean')
        mean = self.repeat(mean, batch_list)
        sub = tensor - mean
        std = segment.segment_reduce(batch_list, sub.pow(2), reducer='mean')
        std = (std + 1e-6).sqrt()
        std = self.repeat(std, batch_list)
        rst = sub / std

        if self.affine:
            rst = self.weight*rst + self.bias
        else:
            rst = rst
        return rst
