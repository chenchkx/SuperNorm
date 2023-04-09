
import torch.nn as nn

class LocalActivation(nn.Module):
    def __init__(self, activation='relu'):
        super(LocalActivation, self).__init__()

        self.activation = activation
        self.activate = None
        if activation == 'relu':
            self.activate = nn.ReLU()
    
    def forward(self, tensor):
        if self.activation == 'None':
            return tensor
        else:
            return self.activate(tensor)

