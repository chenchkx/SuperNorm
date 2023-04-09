
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.norm.norms import NormalizeGNN
from modules.activation.local_activation import LocalActivation
from modules.predict.predict import NodePredict
from modules.convlayer.sgc_layer import SGConvLayer
from modules.pool.global_pool import GlobalPooling
from modules.predict.predict import GraphPredict


### GCN Network for Node Property Prediction
class SGC_IMDB(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer, args):
        super(SGC_IMDB, self).__init__()
        self.num_layer = num_layer
        self.args = args
        
        # middle convolution and norm layers
        self.middle_conv = SGConvLayer(embed_dim, embed_dim, k=self.num_layer,
                                        norm=NormalizeGNN(args.norm_type, embed_dim, affine=args.norm_affine))
        self.middle_norm = NormalizeGNN(args.norm_type, embed_dim, affine=args.norm_affine)

        # predict layer
        self.predict = GraphPredict(embed_dim, output_dim)
   
        # other modules in GNN
        self.activation = LocalActivation(args.activation)
        self.dropout = nn.Dropout(args.dropout)
        self.pooling = GlobalPooling(args.pool_type)
        
    def forward(self, graph, nfeat, efeat=None):

        # initializing node features h_n and edge features h_e
        h_n = F.dropout(nfeat, p=self.args.dropout*self.args.init_dp, training=True)

        h_n = self.middle_conv(graph, h_n)
        h_n = self.middle_norm(graph, h_n)

        h_n = self.dropout(self.activation(h_n))

        # prediction
        g_n = self.pooling(graph, h_n)
        pre = self.predict(g_n)
        return pre
