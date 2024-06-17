# Importing required libraries
import torch
from torch import nn
from torch.nn import functional as F


# Creating VAE-LSTM model class
class VAE_LSTM(nn.Module):
    def __init__(self, configs, device='cuda', num_layers=1):
        super().__init__()

        # Specifying dimensions
        self.input_size = configs.input_size
        self.hidden_size = configs.hidden_size
        self.latent_size = configs.latent_size
        self.output_size = configs.output_size
        self.device = device
        self.num_layers = num_layers

        # Encoder layers
        self.lstm_enc = nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=False)
        self.mean = nn.Linear(self.hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.hidden_size, self.latent_size)

        # Decoder layers
        self.lstm_dec = nn.LSTM(input_size=self.latent_size,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    # For applying re-parameterization trick
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mean + eps * std

    def loss_function(self, *args, **kwargs) -> dict:

        # Defining loss function arguments
        recons = args[0]
        target = args[1]
        mean = args[2]
        logvar = args[3]

        # Weighting the minibatch samples
        kld_weight = 0.00025

        # Calculating reconstruction loss
        recons_loss = F.mse_loss(recons, target)

        # Calculating KL divergence loss
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mean**2 - logvar.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    # Propagating through the network
    def forward(self, x_input):
        batch_size, num_steps, feature_dim = x_input.shape

        # Encoder operations
        out_enc, (h_enc, c_enc) = self.lstm_enc(x_input)
        h_enc_view = h_enc.view(batch_size, self.hidden_size).to(self.device)

        # Latent space representation
        mean = self.mean(h_enc_view)
        logvar = self.logvar(h_enc_view)
        z = self.reparameterize(mean, logvar)

        # Decoder operations
        z = z.repeat(1, num_steps, 1)
        z = z.view(batch_size, num_steps, self.latent_size).to(self.device)

        # Reconstructing output
        out_dec, (h_dec, c_dec) = self.lstm_dec(z, (h_enc, c_enc))
        x_recon = self.out(out_dec)

        # Calculating VAE-LSTM loss
        losses = self.loss_function(x_recon, x_input, mean, logvar)

        calc_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )

        return calc_loss, x_recon, (recon_loss, kld_loss)

