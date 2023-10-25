import torch.nn as nn
import torch.nn.functional as F
import torch


class FinetuneHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        super(FinetuneHead, self).__init__()
        hidden_dim = 256
        prev_dim = input_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else output_dim
            new_layer = nn.Linear(prev_dim, next_dim)
            self.layers.append(new_layer)
            prev_dim = next_dim
        self._init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def _init_weights(self):
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)


