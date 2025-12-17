"""
This module defines the pytorch lightning model for the docking DiT.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

# custom imports:
from custom.model.pLM import calculate_esm_embedding
from custom.data_processing.data import (
    center_and_rotate_chains,
    prepare_input_features,
    scale_coords,
    reverse_scale_coords,
    extract_full_dimer_coords,
    merge_chains,
    split_chains,
)
from custom.utils.inspect import cif_from_tensor

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
            backbone_only,
            test_rmsd = None,
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

        self.sampler = sampler(model=self)

        self.backbone_only = backbone_only

        self.test_rmsd = test_rmsd # MeanRMSD object from custom.utils.benchmarking


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
    

    def on_fit_start(self):
        # assert self.trainer.world_size == 4, "ERROR: 4 gpus enforced."
        print(
            f"[global_rank={self.global_rank}] "
            f"device={self.device}, "
            f"world_size={self.trainer.world_size}",
            flush=True,
        )


    def forward(self, t, feats):
        return self.architecture(t, feats)


    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Performs GPU transformations on the batch of all dataloaders.
        This does not introduce multiplicity yet, because multiplicity can vary between dataloaders, e.g. for validation sampling.
        """
        batch = scale_coords(batch, self.scale_true_coords, self.scale_ref_coords)
        batch = calculate_esm_embedding(batch, self.esm_model, self.esm_alphabet, self.esm_layers)
        return batch
    

    def _shared_step(self, batch, batch_idx):
        with torch.no_grad():
            batch = center_and_rotate_chains(batch, multiplicity=self.multiplicity, device=self.device) # Adds augmented coords for each chain independently
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

            dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x_t)

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

        self.log("trainer/global_step", float(self.global_step), on_step=True, on_epoch=False, logger=True)

        return loss
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Validation loss:
        loss = self._shared_step(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity, add_dataloader_idx=False)
        return loss
    

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        pred_coords = self.predict(batch, batch_idx) # (B, N_atoms_full_dimer, 3)

        assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
        dimer_feat_dict = batch[0]
        true_coords = merge_chains(dimer_feat_dict, "true_coords") # (B, N_atoms_full_dimer, 3)

        self.test_rmsd.update(pred_coords*self.scale_true_coords, true_coords*self.scale_true_coords)
        
        self.log(
            "test/rmsd(A)",
            self.test_rmsd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return None
    

    @torch.no_grad()
    def predict(self, batch, batch_idx):
        """
        Runs sampling on a batch of length 1, i.e. (one dimer per GPU).
        """
        multiplicity = 1
        batch = center_and_rotate_chains(batch, multiplicity=multiplicity, device=self.device) # Adds augmented coords for each chain independently
        assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
        dimer_feat_dict = batch[0]

        embedding_present = all("pLM_emb" in chain_dict for chain_dict in dimer_feat_dict.values())
        assert embedding_present, "ERROR: pLM embedding not present in the chain dictionary."

        dimer_feat_dict = prepare_input_features(dimer_feat_dict, multiplicity, device=self.device)

        dimer_feat_dict = self.sampler.sample(dimer_feat_dict, multiplicity)

        root_dir = self.trainer.default_root_dir
        protein_id = list(dimer_feat_dict.values())[0]["protein_id"]
        file = f"{root_dir}/samples/{protein_id}_pred.cif"
        file_true = f"{root_dir}/samples/{protein_id}_true.cif"

        chain_coords = []
        chain_ids = []
        seqs = []
        for chain_id, chain_dict in dimer_feat_dict.items():
            chain_coords.append(chain_dict["final_sampled_coords"])
            chain_ids.append(chain_id) # IMPORTANT: Biotite only supports 4 letter chain_ids.
            seqs.append(chain_dict["sequence"]) 
        cif_from_tensor(
            chain_coords,
            chain_ids,
            seqs,
            file,
            backbone_only=self.backbone_only,
            scale=self.scale_true_coords
        )

        chain_coords_true = []
        for chain_id, chain_dict in dimer_feat_dict.items():
            chain_coords_true.append(chain_dict["true_coords"])
        cif_from_tensor(
            chain_coords_true,
            chain_ids,
            seqs,
            file_true,
            backbone_only=self.backbone_only,
            scale=self.scale_true_coords
        )

        pred_coords = merge_chains(dimer_feat_dict, "final_sampled_coords") # (B, N_atoms_full_dimer, 3)
        return pred_coords