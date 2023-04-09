
import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling
from modules.pool.dke_pool import DKEPooling

class GlobalPooling(nn.Module):
    def __init__(self, pooling_type):
        super(GlobalPooling, self).__init__()

        self.pooling_type = pooling_type
        if pooling_type == "dke":
            self.pooling = DKEPooling()
        elif pooling_type == "sum":
            self.pooling = SumPooling()
        elif pooling_type == "mean":
            self.pooling = AvgPooling()
        elif pooling_type == "max":
            self.pooling = MaxPooling()

    def forward(self, graph, feat):

        return self.pooling(graph, feat)
