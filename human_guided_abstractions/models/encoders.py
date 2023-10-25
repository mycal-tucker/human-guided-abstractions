import torch.nn as nn
import torch.nn.functional as F


class MNISTEnc(nn.Module):
    def __init__(self, enc_dim):
        super(MNISTEnc, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.fc1 = nn.Linear(3 * 3 * 32, enc_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3 * 3 * 32)
        x = self.fc1(x)
        return x
