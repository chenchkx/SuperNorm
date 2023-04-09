
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder.encoder import NodeEncoder, EdgeEncoder
from modules.convlayer.gin_layer import GINConvLayer
from modules.norm.norms import NormalizeGNN
from modules.activation.local_activation import LocalActivation
from modules.pool.global_pool import GlobalPooling
from modules.predict.predict import GraphPredict
from modules.skip.skip import SkipConnectionLayer

### GIN Network for Graph Property Prediction
class GIN_Graph(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer, args, aggregator_type='sum'):
        super(GIN_Graph, self).__init__()
        self.num_layer = num_layer
        self.args = args
        # input layer
        self.atom_encoder = NodeEncoder(args.dataset_name, embed_dim)
        self.bond_encoder = EdgeEncoder(args.dataset_name, embed_dim)
        # middle layer. i.e., convolutional layer
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        self.skip_layers = nn.ModuleList()
        for i in range(num_layer):
            self.conv_layers.append(GINConvLayer(embed_dim, embed_dim, econv=args.econv, mlp_apply=True))
            self.norm_layers.append(NormalizeGNN(args.norm_type, embed_dim, affine=args.norm_affine))
            if args.skip_type in ['Residual', 'Initial', 'Dense']:
                self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim, 
                                                            layers=i+2, aggregation='concat'))
        if args.skip_type in ['Jumping']:
            self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim, 
                                    layers=num_layer+1, aggregation='concat'))
        
        # predict layer
        self.predict = GraphPredict(embed_dim, output_dim, hidden_dim=embed_dim//2)
   
        # other modules in GNN
        self.activation = LocalActivation(args.activation)
        self.dropout = nn.Dropout(args.dropout)
        self.pooling = GlobalPooling(args.pool_type)


    def forward(self, graph, nfeat, efeat=None):
        # initializing node features h_n and edge features h_e
        h_n = self.dropout(self.activation(self.atom_encoder(nfeat)))
        if efeat is not None:
            h_e = self.dropout(self.activation(self.bond_encoder(efeat)))
        else:
            h_e = efeat

        h_list = [h_n]
        for layer in range(self.num_layer):

            # conv_layer & norm layer
            h_n = self.conv_layers[layer](graph, h_n, h_e)
            h_n = self.norm_layers[layer](graph, h_n)
            # activation and dropout
            h_n = self.activation(h_n)
            h_n = self.dropout(h_n)    
            # skip connection
            if self.args.skip_type in ['Jumping','Residual','Initial', 'Dense']:
                h_list.append(h_n)
            if self.args.skip_type in ['Residual','Initial', 'Dense']:       
                h_n, h_list = self.skip_layers[layer](h_list)
        if self.args.skip_type in ['Jumping']:
            h_n, h_list = self.skip_layers[0](h_list)

        # pooling & prediction
        g_n = self.pooling(graph, h_n)
        pre = self.predict(g_n)
        return pre

