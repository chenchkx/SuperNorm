

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.ops.segment import segment_reduce as sg_reduce
from utils.utils_practice import repeat_tensor_interleave as sg_repeat

class SuperNorm(nn.BatchNorm1d):
    def __init__(self, num_features, edge_power=False, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(SuperNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.edge_power = edge_power
        self.centerScale = nn.BatchNorm1d(num_features, affine=False) # Affine is set to False 

        self.calibration = nn.Parameter(torch.zeros(num_features))
        self.enhancement = nn.Parameter(torch.zeros(num_features))

    def prepare_info(self, graph, tensor): 
        nums = graph.batch_num_nodes()
        tensor = F.normalize(tensor, dim=1) if len(nums) == 1 else tensor    
        calib_factor = graph.ndata['sg_factor_norm']*graph.ndata['sg_factor'] 
        enhan_factor = calib_factor/sg_repeat(sg_reduce(nums, calib_factor, reducer='sum'), nums)
        return nums, tensor, calib_factor, enhan_factor

    def forward(self, graph, tensor):  
        
        nums, tensor, calib_factor, enhan_factor = self.prepare_info(graph, tensor)

        #--- calibration  ---
        cali_tensor = tensor + self.calibration*calib_factor*sg_repeat(sg_reduce(nums, tensor, reducer='mean'), nums)

        cens_tensor = self.centerScale(cali_tensor)
        
        #--- enhancement  ---
        enhan_weight = torch.pow(enhan_factor.repeat(1,self.num_features), self.enhancement)

        if self.affine:    
            results = 0.5*(self.weight+enhan_weight)*cens_tensor+self.bias 
        else:
            results = enhan_weight*cens_tensor    
        return results