import torch
import torch.nn as nn

import human_guided_abstractions.settings as settings
from human_guided_abstractions.models.network_utils import gumbel_softmax
import numpy as np
import torch.nn.functional as F


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        self.prototypes.data.uniform_(-1, 1)
        self.kl_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.projected_prototypes = self.prototypes

    def forward(self, latents):
        protos = self.projected_prototypes if not self.training else self.prototypes
        vector_diffs = latents.unsqueeze(1) - protos
        normalized_dists = torch.sum(vector_diffs ** 2, dim=2)
        neg_dists = -1 * normalized_dists
        onehot, logprobs = gumbel_softmax(neg_dists, hard=True, return_dist=True)
        quantized_latents = torch.matmul(onehot, protos)

        # Compute capacity as the KL divergence from the prior.
        epsilon = 0.0000001
        true_prior = torch.mean(logprobs.exp() + epsilon, dim=0, keepdim=True)
        # Renormalize after adding the epsilon.
        true_prior = true_prior / torch.sum(true_prior, 1)
        prior = true_prior
        prior = prior.expand(logprobs.shape[0], -1)
        complexity = self.kl_loss_fn(logprobs, prior)
        total_loss = settings.kl_weight * complexity

        # Penalize the entropy of the prior, just to reduce codebook size
        ent = torch.sum(-1 * true_prior[0] * true_prior[0].log())  # Just the first row, because it's repeated for batch size.

        total_loss += settings.entropy_weight * ent
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())  # Move prototypes to be near embeddings
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)  # Move embeddings to be near prototypes
        total_loss += embedding_loss + 0.25 * commitment_loss
        return quantized_latents, total_loss

"""
VQ-VIB_C model architecture computes the l2 distance to each quantized vector and samples from the categorical
distribution of prototypes with

P(vq_i|z) \propto exp(-1 * (z -  vq_i) ** 2)
"""
class VQVIBC(nn.Module):
    def __init__(self, output_dim, num_protos, feature_extractor, num_simultaneous_tokens=1):
        super(VQVIBC, self).__init__()
        self.output_dim = output_dim
        self.proto_latent_dim = int(self.output_dim / num_simultaneous_tokens)
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.feature_extractor = feature_extractor
        self.vq_layer = VQLayer(num_protos, self.proto_latent_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        # This reshaping handles the desire to have multiple tokens in a single message.
        reshaped = torch.reshape(features, (-1, self.proto_latent_dim))
        output, total_loss = self.vq_layer(reshaped)
        # Regroup the tokens into messages, now with possibly multiple tokens.
        output = torch.reshape(output, (-1, self.output_dim))

        return output, features, total_loss
