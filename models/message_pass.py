import torch
import torch.nn as nn
from models.layers import Attention


class MPNN(nn.Module):
    def __init__(self,n_layers,edge_network_layers,node_network_layers):

        super().__init__()
        

        self.edge_networks = nn.ModuleList()
        self.node_networks = nn.ModuleList()

        


        
        for i in range(n_layers):
            self.edge_networks.append( MPNN_MLP(edge_network_layers ) )
            self.node_networks.append( MPNN_MLP(node_network_layers ) )
           

     

    def forward(self, node_features, edge_features):
        # node_features B,N,C
        # edge_features B, N,N
        n = node_features.shape[1]

        for enet, nnet in zip(self.edge_networks,self.node_networks):
            u = node_features

            m1 = u.unsqueeze(1).repeat(1, n, 1, 1)  # broadcast to rows
            m2 = u.unsqueeze(2).repeat(1, 1, n, 1)  # broadcast to cols
            m3 = torch.sum(u, dim=1, keepdim=True).unsqueeze(2).repeat(1, n, n, 1)

            block = torch.cat((m1, m2, m3, edge_features), dim=3)

            edge_message = enet(block)
            sum_message = torch.sum(edge_message,dim=2)

            node_input = torch.cat([u,sum_message],dim=2)

            node_features = nnet(node_input)

            node_features = node_features / torch.norm(node_features, p='fro', dim=2, keepdim=True)

        return node_features


class MPNN_MLP(nn.Module):
    def __init__(self, features):

        super().__init__()

        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Linear(features[i],features[i+1]))
            if i < len(features)-2:
                layers.append(nn.ReLU())


        self.net = nn.Sequential(*layers)

    def forward(self, x):
        
        return self.net(x)


