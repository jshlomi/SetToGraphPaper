import torch
import torch.nn as nn
from models.layers import Attention


class RNNmodel(nn.Module):
    def __init__(self, in_features, feats, attention, cfg=None):

        super(RNNmodel, self).__init__()
        if cfg is None:
            cfg = {}

        layers = []

        layers.append(RNN_Layer(in_features, feats[0]))
        for i in range(1, len(feats)):
            layers.append(nn.ReLU())
            layers.append(RNN_Layer(feats[i-1], feats[i]))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequential(x)


class RNN_Layer(nn.Module):
    def __init__(self, in_features, out_features):

        super(RNN_Layer, self).__init__()

        self.gru = nn.GRU(in_features,out_features//2,bidirectional=True,batch_first=True) 
                    


    def forward(self, x):
        # x.shape = (B,C,N)

        x_T = x.transpose(2,1) # (B,N,C)
        x = self.gru(x_T)[0] # (B,N,C_out)
        x = x.transpose(1,2) # (B,C_out,N)
        # normalization
        x = x / torch.norm(x, p='fro', dim=1, keepdim=True)  # BxCxN / Bx1xN

        return x
