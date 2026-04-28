"""
MVP: Adaptive Kalman gain + Information Bottleneck for iPLAN behavior module.

Theory:
  (A) iPLAN Eq.(3) EMA = constant-gain Kalman filter. We make the gain
      data-dependent: eta_t = sigmoid(MLP(log sigma_q^2)).
  (B) HiP-POSG hidden parameter should be the minimum sufficient statistic
      (Tishby 1999). We add VIB KL(q || N(0,I)) with free bits.
"""
import math
import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    return mu + std * torch.randn_like(std)


def kl_standard_normal(mu, logvar, free_bits=0.0):
    # Closed form: KL(N(mu, sigma^2) || N(0, I)) per dim.
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    return kl


class AdaptiveEtaMLP(nn.Module):
    """eta_t = sigmoid(MLP(log sigma_q^2)).

    Weight zero + bias = logit(init_eta) so forward starts at a constant `init_eta`
    matching iPLAN's `soft_update_coef`. The MLP learns to *deviate* from this
    constant as logvar becomes informative, not to *recover* from a neutral 0.5.
    """
    def __init__(self, latent_dim, hidden_dim=16, init_eta=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(init_eta / (1.0 - init_eta)))

    def forward(self, logvar):
        return torch.sigmoid(self.net(logvar))
