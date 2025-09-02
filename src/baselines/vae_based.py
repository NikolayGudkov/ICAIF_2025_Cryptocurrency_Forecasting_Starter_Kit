import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class TinyTimeVAE(nn.Module):
    """
    A tiny VAE-style model for sequence-to-sequence forecasting.
    Encoder: GRU -> latent (mu, logvar)
    Decoder: GRU initialized from latent, predicts 10-step close.
    Inputs: (B, 60, 2)  -> close, volume
    Outputs: (B, 10)    -> close predictions
    """
    def __init__(self, input_dim=2, hidden=64, latent=16, input_len=60, horizon=10):
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.enc_gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

        self.latent_to_h = nn.Linear(latent, hidden)
        self.dec_gru = nn.GRU(1, hidden, batch_first=True)  # decode on previous close
        self.out = nn.Linear(hidden, 1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h = self.enc_gru(x)   # h: (1,B,H)
        h = h.squeeze(0)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, last_close: torch.Tensor) -> torch.Tensor:
        # last_close: (B,) -> start token
        h0 = torch.tanh(self.latent_to_h(z)).unsqueeze(0)  # (1,B,H)
        y = last_close.view(-1, 1, 1)  # (B,1,1)
        outs = []
        for _ in range(self.horizon):
            out, h0 = self.dec_gru(y, h0)
            y = self.out(out)  # (B,1,1)
            outs.append(y.squeeze(-1))  # (B,1)
        return torch.cat(outs, dim=1)  # (B, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B,60,2)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        last_close = x[:, -1, 0]  # use last close as start
        pred = self.decode(z, last_close)  # (B,10)
        return pred, mu, logvar

def vae_loss(pred: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3):
    recon = F.mse_loss(pred, target)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon.item(), kld.item()

def predict_greedy(model: TinyTimeVAE, x: np.ndarray, device="cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(x).float().to(device)
        y, _, _ = model(xt)
    return y.cpu().numpy()