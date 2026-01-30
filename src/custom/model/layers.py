"""
Additional layers for the architecture.
"""


import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class RigidMotionLayer(nn.Module):
    """
    Rigid motion layer. Produces translation vector and angular velocity vector for a single monomer.
    Inputs: Aggregated latent representation from the multimer trunk (B, d_a)
    """

    def __init__(self, d_a, c_dim):
        super().__init__()
        self.d_a = d_a
        self.c_dim = c_dim

        out_channels = 6 # 3 translation + 3 angular velocity

        self.norm_final = nn.LayerNorm(d_a, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(c_dim, 2 * d_a, bias=True)
        )
        self.linear_1 = nn.Linear(d_a, d_a, bias=True)
        self.silu = nn.SiLU()
        self.linear_2 = nn.Linear(d_a, out_channels, bias=True)

        # Initialize output velocity to zero for stability:
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x, c):
        B, d_a = x.shape
        assert d_a == self.d_a, "Shape mismatch"
        assert c.shape == (B, self.c_dim), "Shape mismatch"

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear_1(x)
        x = self.silu(x)
        x = self.linear_2(x) # (B, out_channels) with any real values

        translation = x[:, :3] # (B, 3) with any real values
        angular_velocity = x[:, 3:] # (B, 3) with any real values

        return translation, angular_velocity