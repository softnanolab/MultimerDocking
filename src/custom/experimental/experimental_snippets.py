

########################################################
####    Rigid alignment for flow matching loss     #####
########################################################
"""
Drop in at lightning module training step.
"""
from simplefold.utils.boltz_utils import weighted_rigid_align
with torch.no_grad(), torch.autocast("cuda", enabled=False):
            B, N_atoms, _ = x_1.shape
            v_t = v_predicted.detach().float()
            x_hat = x_t.detach().float() + v_t * (1.0 - t[:, None, None])
            true_coords = x_1.detach().float()
            align_weights = torch.ones(B, N_atoms, device=self.device)
            mask = torch.ones(B, N_atoms, device=self.device)
            x_1_aligned = weighted_rigid_align(
                true_coords,
                x_hat.detach().float(),
                align_weights.float(),
                mask=mask.float(),
            )
            v_target = self.interpolant.velocity_target(t.view(-1, 1, 1), x_0, x_1_aligned, z) # (B, N_atoms_A + N_atoms_B, 3)