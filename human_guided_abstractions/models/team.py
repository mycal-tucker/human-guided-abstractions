import torch.nn as nn


class Team(nn.Module):
    def __init__(self, encoder, listener, decoder):
        super(Team, self).__init__()
        self.encoder = encoder
        self.listener = listener
        self.decoder = decoder

    def forward(self, input_x):
        true_enc, raw_enc, enc_loss = self.encoder(input_x)
        recons = self.decoder(true_enc)
        prediction = self.listener(true_enc)
        return prediction, recons, enc_loss

    def reconstruct(self, encs):
        return self.decoder(encs)
