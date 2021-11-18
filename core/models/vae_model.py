import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Any
from torch.nn import functional as F


class VanillaVAE(nn.Module):
    """
    Vanilla Variational Auto Encoder model.

    :Interfaces: encode, decode, reparameterize, forward, loss_function, sample, generate

    :Arguments:
        - in_channels (int): the channel number of input
        - latent_dim (int): the latent dimension of the middle representation
        - hidden_dims (List): the hidden dimensions of each layer in the MLP architecture in encoder and decoder
        - kld_weight(float): the weight of KLD loss
    """

    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            hidden_dims: List = None,
            kld_weight: float = 0.1,
    ) -> None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims
        self.kld_weight = kld_weight

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim), nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # original fc
        self.fc_mu = nn.Linear(hidden_dims[-1] * 36, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 36, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 36)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1
                    ), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=7, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.

        :Arguments:
            - input (Tensor): Input tensor to encode [N x C x H x W]
        :Returns:
            Tensor: List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.

        :Arguments:
            - z (Tensor): [B x D]
        :Returns:
            Tensor: Output decode tensor [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 6, 6)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        :Arguments:
            - mu (Tensor): Mean of the latent Gaussian [B x D]
            - logvar (Tensor): Standard deviation of the latent Gaussian [B x D]
        :Returns:
            Tensor: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        [summary]

        :Arguments:
            - input (torch.Tensor): Input tensor

        :Returns:
            List[torch.Tensor]: Input and output tensor
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        #z = mu
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> Dict:
        """
        Computes the VAE loss function.

        :math:`KL(N(\mu, \sigma), N(0, 1)) = \log \\frac{1}{\sigma} + \\frac{\sigma^2 + \mu^2}{2} - \\frac{1}{2}`

        :Returns:
            Dict: Dictionary containing loss information
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        #kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        kld_weight = self.kld_weight

        recons_loss = 0
        '''
        weight = [8.7924e-01, 7.4700e-02, 1.0993e-02, 6.1075e-04, 2.6168e-03, 2.8066e-02, 3.7737e-03]
        vd = 1
        for i in range(7):
            cur = F.l1_loss(recons[:, i, ...], input[:, i, ...])
            recons_loss += 1 / weight[i] * cur * vd
            ret[str(i)] = cur
            if i==0 and cur > 0.05:
                vd = 0
        '''

        recons_loss = F.mse_loss(recons, input)
        if recons_loss < 0.05:
            recons_loss = F.l1_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> torch.Tensor:
        r"""
        Samples from the latent space and return the corresponding
        image space map.

        :Arguments:
            - num_samples(Int): Number of samples.
            - current_device(Int): Device to run the model.
        :Returns:
            Tensor: Sampled decode tensor.
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image

        :Arguments:
            - x(Tensor): [B x C x H x W]
        :Returns:
            Tensor: [B x C x H x W]
        """

        return self.forward(x)[0]
