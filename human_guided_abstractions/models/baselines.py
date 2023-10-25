import torch
import torch.nn as nn
from human_guided_abstractions.models.network_utils import reparameterize
import human_guided_abstractions.settings as settings


class VAE(nn.Module):
    def __init__(self, output_dim, feature_extractor):
        super().__init__()
        self.output_dim = output_dim
        self.feature_extractor = feature_extractor
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_logvar = nn.Linear(output_dim, output_dim)
        self.num_tokens = -1

    def forward(self, x):
        features = self.feature_extractor(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        z = reparameterize(mu, logvar)
        divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return z, mu, settings.kl_weight * divergence
