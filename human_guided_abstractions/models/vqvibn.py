import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import human_guided_abstractions.settings as settings
from human_guided_abstractions.models.network_utils import reparameterize


class VQLayer(nn.Module):
    def __init__(self, num_protos, latent_dim, beta=0.25):
        super(VQLayer, self).__init__()
        self.num_protos = num_protos
        self.latent_dim = latent_dim
        self.beta = beta
        self.prototypes = nn.Parameter(data=torch.Tensor(num_protos, latent_dim))
        self.prototypes.data.uniform_(-1 / self.num_protos, 1 / self.num_protos)
        self.projected_prototypes = self.prototypes

    def forward(self, latents, mus=None):
        protos = self.projected_prototypes if not self.training else self.prototypes
        dists_to_protos = torch.sum(latents ** 2, dim=1, keepdim=True) + \
                          torch.sum(protos ** 2, dim=1) - 2 * \
                          torch.matmul(latents, protos.t())
        closest_protos = torch.argmin(dists_to_protos, dim=1).unsqueeze(1)
        encoding_one_hot = torch.zeros(closest_protos.size(0), self.num_protos).to(settings.device)
        encoding_one_hot.scatter_(1, closest_protos, 1)
        quantized_latents = torch.matmul(encoding_one_hot, protos)

        # Compute the VQ Losses
        if mus is None:
            commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
            embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        else:
            commitment_loss = F.mse_loss(quantized_latents.detach(), mus)
            embedding_loss = F.mse_loss(quantized_latents, mus.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss
        # Compute the entropy of the distribution for which prototypes are used. Uses a differentiable approximation
        # for the distributions.
        epsilon = 0.00000001
        vector_diffs = mus.unsqueeze(1) - protos
        normalized_diffs = vector_diffs
        square_distances = torch.square(normalized_diffs)
        normalized_dists = torch.sum(square_distances, dim=2)
        neg_dist = -1.0 * normalized_dists
        exponents = neg_dist.exp() + epsilon
        row_sums = torch.sum(exponents, dim=1, keepdim=True)
        row_probs = torch.div(exponents, row_sums)
        approx_probs = torch.mean(row_probs, dim=0)
        logdist = approx_probs.log()
        ent = torch.sum(-1 * approx_probs * logdist)
        vq_loss += settings.entropy_weight * ent

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents, vq_loss


"""
VQ-VIB_N model architecture samples from a Gaussian and then discretizes.
"""
class VQVIBN(nn.Module):
    def __init__(self, output_dim, num_protos, feature_extractor, num_simultaneous_tokens=1):
        super(VQVIBN, self).__init__()
        self.output_dim = output_dim
        self.proto_latent_dim = int(self.output_dim / num_simultaneous_tokens)
        self.num_tokens = num_protos  # Need this general variable for num tokens
        self.num_simultaneous_tokens = num_simultaneous_tokens
        self.feature_extractor = feature_extractor

        self.vq_layer = VQLayer(num_protos, self.proto_latent_dim)
        self.fc_mu = nn.Linear(self.output_dim, self.output_dim)
        self.fc_var = nn.Linear(self.output_dim, self.output_dim)


    def forward(self, x):
        features = self.feature_extractor(x)
        logvar = self.fc_var(features)
        mu = self.fc_mu(features)
        sample = reparameterize(mu, logvar)
        reshaped = torch.reshape(sample, (-1, self.proto_latent_dim))
        reshaped_mu = torch.reshape(mu, (-1, self.proto_latent_dim))
        reshaped_logvar = torch.reshape(logvar, (-1, self.proto_latent_dim))
        # Quantize the vectors
        discretized, quantization_loss = self.vq_layer(reshaped, reshaped_mu)
        discretized = torch.reshape(discretized, (-1, self.output_dim))
        # Compute the KL divergence
        divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        # Total loss is the penalty on complexity plus the quantization (and entropy) losses.
        total_loss = settings.kl_weight * divergence + quantization_loss
        return discretized, sample, total_loss
