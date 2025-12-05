"""
This module defines the pytorch lightning model for the docking DiT.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

# custom imports:
from custom.model.pLM import esm_model_dict, calculate_esm_embedding
from custom.data_processing.data import (
    center_and_rotate_chains,
    prepare_input_features,
    scale_coords,
    extract_full_dimer_coords,
    merge_chains,
    split_chains,
)


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
            multiplicity,
            scale_true_coords,
            scale_ref_coords,
            interpolant,
            optimizer,
            scheduler,
            sampler,
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
        
        self.multiplicity = multiplicity
        
        self.scale_true_coords = scale_true_coords
        self.scale_ref_coords = scale_ref_coords
        
        self.interpolant = interpolant
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.sampler = sampler(model=self, multiplicity=self.multiplicity)

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
    
    def forward(self, t, feats):
        return self.architecture(t, feats)

    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Performs GPU transformations on the batch of all dataloaders.
        """
        batch = scale_coords(batch, self.scale_true_coords, self.scale_ref_coords)
        batch = calculate_esm_embedding(batch, self.esm_model, self.esm_alphabet, self.esm_layers)
        batch = center_and_rotate_chains(batch, multiplicity=self.multiplicity, device=self.device) # Adds augmented coords for each chain independently
        return batch
    
    def _shared_step(self, batch, batch_idx):
        with torch.no_grad():
            assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
            dimer_feat_dict = batch[0]
            dimer_feat_dict = prepare_input_features(dimer_feat_dict, self.multiplicity, device=self.device)

            # Sample timesteps:
            t_size = self.multiplicity * len(batch)
            t = 0.98 * logit_normal_sample(n=t_size, m=0.8, s=1.7) + 0.02 * torch.rand(
                t_size
            )
            t = t.to(self.device)

            x_0 = merge_chains(dimer_feat_dict, "augmented_coords") # (B, N_atoms_A + N_atoms_B, 3)
            x_1 = extract_full_dimer_coords(dimer_feat_dict, device=self.device) # (B, N_atoms_A + N_atoms_B, 3)
            z = torch.randn_like(x_1) # (B, N_atoms_A + N_atoms_B, 3)
            x_t = self.interpolant.compute_x_t(t.view(-1, 1, 1), x_0, x_1, z) # (B, N_atoms_A + N_atoms_B, 3)

            dimer_feat_dict = split_chains(dimer_feat_dict, "augmented_coords", x_t)

        # Forward pass:
        dimer_feat_dict = self.forward(t, dimer_feat_dict)
        v_predicted = merge_chains(dimer_feat_dict, "velocity_field") # (B, N_atoms_A + N_atoms_B, 3)
        v_target = self.interpolant.velocity_target(t.view(-1, 1, 1), x_0, x_1, z) # (B, N_atoms_A + N_atoms_B, 3)

        loss = F.mse_loss(v_predicted, v_target)
        return loss
    
    def on_train_epoch_start(self):
        print(f"Starting training epoch {self.current_epoch}")
        print(f"Multiplicity: {self.multiplicity}")

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        # Logging:
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        # Logging:
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)

        return loss

    # def on_validation_epoch_end(self):
    #     # Run validation sampling
    #     pass
    
    # @torch.no_grad()
    # def _validation_sampling(self, batch, batch_idx):
    #     """
    #     """
    #     assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
    #     dimer_feat_dict = batch[0]
    #     dimer_feat_dict = prepare_input_features(dimer_feat_dict, self.multiplicity, device=self.device)