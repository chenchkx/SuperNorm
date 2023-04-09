
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.norm.norms import NormalizeGNN
from modules.activation.local_activation import LocalActivation
from modules.pool.global_pool import GlobalPooling
from modules.predict.predict import GraphPredict

# MLP for IMDB Prediction
class MLP_Layer(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, norm_type='batchnorm', dropout=0.5):
        super(MLP_Layer, self).__init__()

        self.input_layer = nn.Linear(input_dim, middle_dim)
        self.norm_layer = NormalizeGNN(norm_type, middle_dim)
        self.relu_layer = nn.ReLU()
        self.output_layer = nn.Linear(middle_dim, output_dim)

    def forward(self, graph, nfeat):

        nfeat = self.input_layer(nfeat)
        nfeat = self.norm_layer(graph, nfeat)
        nfeat = self.relu_layer(nfeat)
        nfeat = self.output_layer(nfeat)

        return nfeat


### MLP Network(Single Layer) for Graph Property Prediction
class MLP_IMDB(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer, args):
        super(MLP_IMDB, self).__init__()
        self.num_layer = num_layer
        self.args = args
        
        self.conv_layer = MLP_Layer(embed_dim, embed_dim, embed_dim, args.norm_type)
        self.norm_layer = NormalizeGNN(args.norm_type, embed_dim, affine=args.norm_affine)

        # predict layer
        self.predict = GraphPredict(embed_dim, output_dim, hidden_dim=embed_dim//2)
   
        # other modules in GNN
        self.activation = LocalActivation(args.activation)
        self.dropout = nn.Dropout(args.dropout)
        self.pooling = GlobalPooling(args.pool_type)

    def forward(self, graph, nfeat, efeat=None):

        h_n = F.dropout(nfeat, p=self.args.dropout*self.args.init_dp, training=True)

        # conv_layer & norm layer
        h_n = self.conv_layer(graph, h_n)
        h_n = self.norm_layer(graph, h_n)
        
        # activation and dropout
        h_n = self.activation(h_n)
        h_n = self.dropout(h_n)    

        # pooling & prediction
        g_n = self.pooling(graph, h_n)
        pre = self.predict(g_n)
        return pre
