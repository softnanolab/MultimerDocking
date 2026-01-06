"""
Additional layers for the architecture.
"""

import torch
import torch.nn as nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class RigidMotionLayer(nn.Module):
    """
    Rigid motion layer. Produces translation vector and angular velocity vector.
    Inputs: Latent from the multimer residue trunk of shape (B, L_r, d_a)
    """

    def __init__(self, L_r, d_a, n_monomers, c_dim=None):
        super().__init__()
        self.L_r = L_r
        self.d_a = d_a
        self.n_monomers = n_monomers


        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        B, L_r, d_a = x.shape
        assert L_r == self.L_r and d_a == self.d_a, "Shape mismatch"

        # Flatten the latent representation:
        x = x.reshape(B, self.n_monomers, self.L_r, self.d_a)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)


        x = self.linear(x)

        return x