"""
This module defines the pytorch lightning model for the docking DiT.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from utils.boltz_utils import weighted_rigid_align

# custom imports:
from custom.model.pLM import esm_model_dict, calculate_esm_embedding
from custom.data_processing.data import center_and_rotate_chains, prepare_input_features, scale_coords, extract_and_center_full_dimer, merge_chains, split_chains


def logit_normal_sample(n=1, m=0.0, s=1.0):
    # Logit-Normal Sampling from https://arxiv.org/pdf/2403.03206.pdf
    u = torch.randn(n) * s + m
    t = 1 / (1 + torch.exp(-u))
    return t


########################################################
####              Lightning Module                 #####
########################################################
class DockingModel(pl.LightningModule):
    def __init__(
            self,
            architecture,
            esm_model,
            esm_layers,
            esm_num_layers,
            esm_d_e,
            t_multiplicity,
            scale_true_coords,
            scale_ref_coords,
            interpolant,
            optimizer,
            scheduler,
    ):
        super().__init__()
        
        self.esm_model, self.esm_alphabet, model_key = esm_model
        # Freeze ESM parameters explicitly:
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.esm_model.eval()
        
        self.esm_layers = esm_layers
        if self.esm_layers == "all":
            self.esm_layers = list(range(esm_num_layers))

        self.esm_d_e = esm_d_e
        
        self.architecture = architecture
        
        self.t_multiplicity = t_multiplicity
        
        self.scale_true_coords = scale_true_coords
        self.scale_ref_coords = scale_ref_coords
        
        self.interpolant = interpolant
        
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, t, feats):
        return self.architecture(t, feats)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = scale_coords(batch, self.scale_true_coords, self.scale_ref_coords)

            # ESM embedding:
            batch = calculate_esm_embedding(batch, self.esm_model, self.esm_alphabet, self.esm_layers)
            
            # Center and randomly rotate chains:
            batch = center_and_rotate_chains(batch, device=self.device)

            # Prepare dimer features for forward pass:
            assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
            dimer_feat_dict = batch[0]

            dimer_feat_dict = prepare_input_features(dimer_feat_dict, self.t_multiplicity, device=self.device)

            # Sample timesteps for forward pass:
            t_size = self.t_multiplicity * len(batch)
            t = 0.98 * logit_normal_sample(n=t_size, m=0.8, s=1.7) + 0.02 * torch.rand(
                t_size
            )
            t = t.to(self.device)
            # What about numerical stability for t~0 or 1?

            x_0 = merge_chains(dimer_feat_dict, "augmented_coords") # (B, N_atoms_A + N_atoms_B, 3)
            x_1 = extract_and_center_full_dimer(dimer_feat_dict, device=self.device) # (B, N_atoms_A + N_atoms_B, 3)
            z = torch.randn_like(x_1) # (B, N_atoms_A + N_atoms_B, 3)
            x_t = self.interpolant.compute_x_t(t.view(-1, 1, 1), x_0, x_1, z) # (B, N_atoms_A + N_atoms_B, 3)

            dimer_feat_dict = split_chains(dimer_feat_dict, "augmented_coords", x_t)

        # Forward pass:
        dimer_feat_dict = self.forward(t, dimer_feat_dict)
        v_predicted = merge_chains(dimer_feat_dict, "velocity_field") # (B, N_atoms_A + N_atoms_B, 3)

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

        loss = F.mse_loss(v_predicted, v_target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch))
        
        return loss

    def configure_optimizers(self):
        # Exclude ESM model parameters since they're frozen:
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optimizer(params=trainable_params)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
