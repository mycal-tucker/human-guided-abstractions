import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Listener, self).__init__()
        hidden_dim = 128
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)
