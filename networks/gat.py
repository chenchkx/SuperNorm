
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder.encoder import NodeEncoder, EdgeEncoder
from modules.convlayer.gat_layer import GATConvLayer
from modules.norm.norms import NormalizeGNN
from modules.activation.local_activation import LocalActivation
from modules.pool.global_pool import GlobalPooling
from modules.predict.predict import GraphPredict, NodePredict
from modules.skip.skip import SkipConnectionLayer

### GAT Network for Graph Property Prediction
class GAT_Graph(nn.Module):
    def __init__(self, embed_dim, output_dim, num_layer, args, num_heads=8):
        super(GAT_Graph, self).__init__()
        self.num_layer = num_layer
        self.args = args
        
        # input layer
        self.atom_encoder = NodeEncoder(args.dataset_name, embed_dim*num_heads)
        self.bond_encoder = EdgeEncoder(args.dataset_name, embed_dim*num_heads)
        # middle layer. i.e., convolutional layer
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        self.skip_layers = nn.ModuleList()
        for i in range(num_layer):
            self.conv_layers.append(GATConvLayer(embed_dim*num_heads, embed_dim, num_heads=num_heads, econv=args.econv))
            self.norm_layers.append(NormalizeGNN(args.norm_type, embed_dim*num_heads, affine=args.norm_affine))
            if args.skip_type in ['Residual', 'Initial', 'Dense']:
                self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim*num_heads, 
                                                            layers=i+2, aggregation='concat'))
        if args.skip_type in ['Jumping']:
            self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim*num_heads, 
                                    layers=num_layer+1, aggregation='concat'))
        
        # predict layer
        self.predict = GraphPredict(embed_dim*num_heads, output_dim, hidden_dim=embed_dim*num_heads//2)
   
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


### GAT Network for Node Property Prediction
class GAT_Node(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, num_layer, args, num_heads=8):
        super(GAT_Node, self).__init__()
        self.num_layer = num_layer
        self.args = args

        # encoders for convolution initialization
        self.node_encoder = GATConvLayer(input_dim, embed_dim, num_heads=num_heads)
        self.edge_encoder = EdgeEncoder(args.dataset_name, embed_dim)

        # middle convolution and norm layers 
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() 
        self.skip_layers = nn.ModuleList()
        for i in range(num_layer):
            self.conv_layers.append(GATConvLayer(embed_dim*num_heads, embed_dim, num_heads=num_heads, econv=args.econv))
            self.norm_layers.append(NormalizeGNN(args.norm_type, embed_dim*num_heads, affine=args.norm_affine))
            if args.skip_type in ['Residual', 'Initial', 'Dense']:
                self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim*num_heads, 
                                                            layers=i+2, aggregation='concat'))
        if args.skip_type in ['Jumping']:
            self.skip_layers.append(SkipConnectionLayer(args.skip_type, embed_dim=embed_dim*num_heads, 
                                    layers=num_layer+1, aggregation='concat'))
        
        # predict layer
        self.predict = NodePredict(embed_dim*num_heads, output_dim)
   
        # other modules in GNN
        self.activation = LocalActivation(args.activation)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, graph, nfeat, efeat=None):

        # initializing node features h_n and edge features h_e
        h_n = self.dropout(self.activation(self.node_encoder(graph, nfeat)))
        if efeat is not None:
            h_e = self.dropout(self.activation(self.edge_encoder(efeat)))
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

        # prediction
        pre = self.predict(h_n)
        return pre
