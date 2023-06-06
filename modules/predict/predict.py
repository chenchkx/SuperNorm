
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.activation.local_activation import LocalActivation

class GraphPredict(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None,
                       activation='relu', dropout=0.5):
        super(GraphPredict, self).__init__()

        if hidden_dim is not None and hidden_dim<=min(input_dim, output_dim):
            hidden_dim = min(input_dim,output_dim)

        if hidden_dim is None:
            self.predict = nn.Linear(input_dim, output_dim)
        else:
            self.predict = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                LocalActivation(activation),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, output_dim)
            )  
           
    def forward(self, tensor):

        return self.predict(tensor)


class NodePredict(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None,
                       activation='relu', dropout=0.5):
        super(NodePredict, self).__init__()

        if hidden_dim is not None and hidden_dim<=min(input_dim, output_dim):
            hidden_dim = min(input_dim,output_dim)

        if hidden_dim is None:
            self.predict = nn.Linear(input_dim, output_dim)
        else:
            self.predict = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                LocalActivation(activation),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, output_dim)
            )  
      
    def forward(self, tensor):

        return self.predict(tensor)



class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, dropout=0.5):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
 
    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)






