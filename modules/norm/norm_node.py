import torch
import torch.nn as nn
import torch.nn.functional as F


class pair_norm(torch.nn.BatchNorm1d):
    def __init__(self, mode='pairnorm', scale=1, num_features=128):
        super(pair_norm, self).__init__(num_features)
        self.mode = mode
        self.scale = scale
        self.bn = nn.BatchNorm1d(num_features)
        self.mean_scale = nn.Parameter(torch.randn(num_features))

    def forward(self, graph, x):
           
        if self.mode == 'pairnorm':
            col_mean = x.mean(dim=0)   
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'pairnorm-gp': #
            batch_list = graph.batch_num_nodes()
            batch_size = len(batch_list)
            batch_index = torch.arange(batch_size).to(x.device).repeat_interleave(batch_list)
            batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
            mean = torch.zeros(batch_size, *x.shape[1:]).to(x.device)
            mean = mean.scatter_add_(0, batch_index, x)
            mean = (mean.T / batch_list).T
            col_mean = mean.repeat_interleave(batch_list, dim=0)
            
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean
            
        return x


class mean_norm(torch.nn.BatchNorm1d):
    def __init__(self, mode='meannorm', num_features=128):
        super(mean_norm, self).__init__(num_features)
        self.mode = mode
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, graph, x):
        
        if self.mode == 'meannorm':
            col_mean = x.mean(dim=0)
            x = x - col_mean

        if self.mode == 'meannorm-gp': #
            batch_list = graph.batch_num_nodes()
            batch_size = len(batch_list)
            batch_index = torch.arange(batch_size).to(x.device).repeat_interleave(batch_list)
            batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
            mean = torch.zeros(batch_size, *x.shape[1:]).to(x.device)
            mean = mean.scatter_add_(0, batch_index, x)
            mean = (mean.T / batch_list).T
            col_mean = mean.repeat_interleave(batch_list, dim=0) 
            
            x = x - col_mean
        if self.mode == 'meannorm-bn': # mean norm with batchnorm
            col_mean = x.mean(dim=0)
            x = self.bn(x - col_mean)

        return x


class node_norm(torch.nn.Module):
    def __init__(self, node_norm_type="n", unbiased=False, eps=1e-5, power_root=2, mode='nodenorm', num_features=128, **kwargs):
        super(node_norm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.node_norm_type = node_norm_type
        self.power = 1 / power_root
        self.bn = nn.BatchNorm1d(num_features)
        self.mode = mode
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        # in GCN+Cora, 
        # n v srv pr
        # 16 layer:  _19.8_  15.7 17.4 17.3
        # 32 layer:  20.3 _25.5_ 16.2 16.3

        if self.node_norm_type == "n":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.node_norm_type == "v":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std

        elif self.node_norm_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.node_norm_type == "srv":  # squre root of variance
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.node_norm_type == "pr":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        if self.mode == 'nodenorm-bn':
            x = self.bn(x)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        node_norm_type_str = f"node_norm_type={self.node_norm_type}"
        components.insert(-1, node_norm_type_str)
        new_str = "".join(components)
        return new_str


class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_groups=None, skip_weight=None, model='groupnorm', **w):
        super(group_norm, self).__init__()
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.model = model

        dim_hidden = dim_hidden if dim_to_norm is None else dim_to_norm
        self.dim_hidden = dim_hidden

        # print(f'\n\n{dim_to_norm}\n\n');raise

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        self.bn_out = nn.BatchNorm1d(dim_hidden)
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=1)
            x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)],
                               dim=1)
            x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)

        x = x + x_temp * self.skip_weight 
        if self.model =='groupnorm-bn':
            x = self.bn_out(x)
        return x

