import torch
from torch import nn


class ResidualConnection(nn.Module):
    def __init__(self, alpha=0.5):
        super(ResidualConnection, self).__init__()
        self.alpha = alpha

    def forward(self, Xs: list):
        assert len(Xs) >= 1
        # if len(Xs) <= 4:
        #     Xs[-1] = Xs[-1]
        # else: 
        #     Xs[-1] = Xs[-1] + Xs[-3]

        if len(Xs)%2 == 1 and len(Xs)>2:
            Xs[-1] = Xs[-1] + Xs[-3]
        else: 
            Xs[-1] = Xs[-1]
        return Xs[-1], Xs 
 

class InitialConnection(nn.Module):
    def __init__(self, alpha=0.5):
        super(InitialConnection, self).__init__()
        self.alpha = alpha

    def forward(self, Xs: list):
        assert len(Xs) >= 1
        if len(Xs) <= 2:
            Xs[-1] = Xs[-1]
        else: 
            Xs[-1] = (1 - self.alpha) * Xs[-1] + self.alpha * Xs[1]
            # Xs[-1] = Xs[-1] + Xs[1]
        return Xs[-1], Xs


class DenseConnection(nn.Module):
    def __init__(self, in_dim, out_dim, aggregation='concat'):
        super(DenseConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregation = aggregation
        if aggregation == 'concat':
            self.layer_transform = nn.Linear(in_dim, out_dim, bias=True)
        elif aggregation == 'attention':
            self.layer_att = nn.Linear(in_dim, 1, bias=True)

    def forward(self, Xs: list):
        assert len(Xs) >= 1
        if self.aggregation == 'concat':
            X = torch.cat(Xs, dim=-1)
            X = self.layer_transform(X)
            return X, Xs
        elif self.aggregation == 'maxpool':
            X = torch.stack(Xs, dim=-1)
            X, _ = torch.max(X, dim=-1, keepdim=False)
            return X, Xs
        # implement with the code from https://github.com/mengliu1998/DeeperGNN/blob/master/DeeperGNN/dagnn.py
        elif self.aggregation == 'attention':
            # pps n x k+1 x c
            pps = torch.stack(Xs, dim=1)
            retain_score = self.layer_att(pps).squeeze()
            retain_score = torch.sigmoid(retain_score).unsqueeze(1)
            X = torch.matmul(retain_score, pps).squeeze()
            return X, Xs
        else:
            raise Exception("Unknown aggregation")


class SkipConnectionLayer(nn.Module):
    def __init__(self, skip_type, embed_dim=128, layers=0, aggregation='concat'):
        super(SkipConnectionLayer, self).__init__()
        self.skip_type = skip_type
        self.skip = None

        if self.skip_type == 'Residual':
            self.skip = ResidualConnection()
        elif self.skip_type == 'Initial':
            self.skip = InitialConnection()
        elif self.skip_type == 'Dense':
            self.skip = DenseConnection((layers)*embed_dim, embed_dim, aggregation)
        elif self.skip_type == 'Jumping':
            self.skip = DenseConnection((layers)*embed_dim, embed_dim, aggregation)

    def forward(self, Xs: list):

        if self.skip_type == 'None':
            X, Xs = Xs[-1], Xs
        else:
            X, Xs = self.skip(Xs)

        return X, Xs 
