"""
This module defines the pytorch lightning model for the docking DiT.
"""

import csv
import os
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import torch.distributed as dist

from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

# custom imports:
from custom.model.pLM import calculate_mint_embedding
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
from custom.utils.structure_utils import aligned_weighted_cross_RMSD, create_crop_mask


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
            mint_model,
            mint_use_weights,
            mint_layers,
            multiplicity,
            scale_true_coords,
            scale_ref_coords,
            interpolant,
            optimizer,
            scheduler,
            sampler,
            backbone_only,
            test_rmsd = None,
            test_single_chain_rmsd = None,
            crop_terminal_residues = None,
            test_dockq = None,
            test_fnat_dockq = None,
            test_iRMSD_dockq = None,
            test_LRMSD_dockq = None,
            test_failing_dockq = None,
            test_cross_chain_rmsd_A = None,
            test_cross_chain_rmsd_B = None,
    ):
        super().__init__()

        ### Start of Mint model preparations:
        self.mint_model = mint_model
        self.mint_layers = mint_layers

        # Load mint weights:
        self.mint_model.load_pretrained_weights(model_name=mint_use_weights)
        
        # Freeze pLM parameters explicitly:
        for param in self.mint_model.parameters():
            param.requires_grad = False
        
        self.mint_model.eval()
        ### End of Mint model preparations.
        
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
        self.test_single_chain_rmsd = test_single_chain_rmsd # SingleChainRMSD object from custom.utils.benchmarking
        self.crop_terminal_residues = crop_terminal_residues
        self.test_dockq = test_dockq # DockQMetric object from custom.utils.benchmarking
        self.test_fnat_dockq = test_fnat_dockq # fnat_DockQMetric object from custom.utils.benchmarking
        self.test_iRMSD_dockq = test_iRMSD_dockq # iRMSD_DockQMetric object from custom.utils.benchmarking
        self.test_LRMSD_dockq = test_LRMSD_dockq # LRMSD_DockQMetric object from custom.utils.benchmarking
        self.test_failing_dockq = test_failing_dockq # FailingDockQMetric object from custom.utils.benchmarking
        self.test_records = []

    def configure_optimizers(self):
        # Exclude pLM model parameters since they're frozen:
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


    def on_test_start(self):
        self.test_records = []


    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)


    def forward(self, t, feats):
        return self.architecture(t, feats)


    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Performs GPU transformations on the batch of all dataloaders.
        This does not introduce multiplicity yet, because multiplicity can vary between dataloaders, e.g. for validation sampling.
        """
        batch = scale_coords(batch, self.scale_true_coords, self.scale_ref_coords)
        batch = calculate_mint_embedding(batch, self.mint_model, self.mint_layers)
        return batch

    def compute_cross_RMSD_loss(self, v_pred, x_t, x_1, t, dimer_feat_dict):
        """
        - v_pred: (B, N_atoms_A + N_atoms_B, 3)
        - x_t: (B, N_atoms_A + N_atoms_B, 3)
        - x_1: (B, N_atoms_A + N_atoms_B, 3)
        - t: (B, 1, 1)
        - dimer_feat_dict: dict
        """
        chain_ids = list(dimer_feat_dict.keys())
        N_atoms_A = dimer_feat_dict[chain_ids[0]]["N_atoms"]
        N_atoms_B = dimer_feat_dict[chain_ids[1]]["N_atoms"]

        # One-shot Euler:
        x_hat = x_t + (1 - t)*v_pred

        # 1 for chain A and 0 for chain B, shape = (B, N_atoms_A + N_atoms_B)
        chain_mask = torch.cat([torch.ones(N_atoms_A, device=self.device), torch.zeros(N_atoms_B, device=self.device)], dim=0).unsqueeze(0).repeat_interleave(x_t.size(0), dim=0) 

        rmsd_A = torch.mean(aligned_weighted_cross_RMSD(x_hat, x_1, chain_mask)) # average over the batch
        rmsd_B = torch.mean(aligned_weighted_cross_RMSD(x_hat, x_1, 1 - chain_mask))
        
        return rmsd_A, rmsd_B


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
            # z = torch.randn_like(x_1) # (B, N_atoms_A + N_atoms_B, 3)
            z = torch.zeros_like(x_1) # Test training with no noise.
            x_t = self.interpolant.compute_x_t(t.view(-1, 1, 1), x_0, x_1, z) # (B, N_atoms_A + N_atoms_B, 3)

            dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x_t)

        # Forward pass:
        dimer_feat_dict = self.forward(t, dimer_feat_dict)
        v_predicted = merge_chains(dimer_feat_dict, "velocity_field") # (B, N_atoms_A + N_atoms_B, 3)
        v_target = self.interpolant.velocity_target(t.view(-1, 1, 1), x_0, x_1, z) # (B, N_atoms_A + N_atoms_B, 3)

        
        loss_flow = F.mse_loss(v_predicted, v_target)
        rmsd_A, rmsd_B = self.compute_cross_RMSD_loss(v_predicted, x_t, x_1, t.view(-1, 1, 1), dimer_feat_dict)

        return loss_flow, rmsd_A, rmsd_B
    

    def on_train_epoch_start(self):
        print(f"Starting training epoch {self.current_epoch}")
        print(f"Multiplicity: {self.multiplicity}")


    def training_step(self, batch, batch_idx):
        loss_flow, rmsd_A, rmsd_B = self._shared_step(batch, batch_idx)

        loss = loss_flow + rmsd_A + rmsd_B

        # Logging:
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)
        self.log("train/loss_flow", loss_flow, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)
        self.log("train/rmsd_A", rmsd_A, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)
        self.log("train/rmsd_B", rmsd_B, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity)

        self.log("trainer/global_step", float(self.global_step), on_step=True, on_epoch=False, logger=True)

        return loss
    

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Validation loss:
        loss_flow, rmsd_A, rmsd_B  = self._shared_step(batch, batch_idx)
        loss = loss_flow + rmsd_A + rmsd_B
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity, add_dataloader_idx=False)
        self.log("val/loss_flow", loss_flow, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity, add_dataloader_idx=False)
        self.log("val/rmsd_A", rmsd_A, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity, add_dataloader_idx=False)
        self.log("val/rmsd_B", rmsd_B, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=len(batch)*self.multiplicity, add_dataloader_idx=False)
        return loss
    

    @torch.no_grad()
    def test_step(self, batch, batch_idx):

        pred_coords, file_pred, file_true = self.predict(batch, batch_idx) # (B, N_atoms_full_dimer, 3)

        assert len(batch) == 1, "ERROR: Only one dimer per GPU allowed."
        dimer_feat_dict = batch[0]
        true_coords = merge_chains(dimer_feat_dict, "true_coords") # (B, N_atoms_full_dimer, 3)

        # Crop M residues from all chain termini:
        crop_mask = create_crop_mask(dimer_feat_dict, M=self.crop_terminal_residues) # (B, N_atoms_A + N_atoms_B)
        
        ######### Dimer RMSD:
        self.test_rmsd.update(
            pred_coords*self.scale_true_coords,
            true_coords*self.scale_true_coords,
            crop_mask.float()
        )
        dimer_rmsd = self.test_rmsd.last_value
        self.log(
            "test/dimer_rmsd(A)",
            self.test_rmsd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        ########

        ######## Chain RMSD:
        chain_ids = list(dimer_feat_dict.keys())
        N_atoms_A = dimer_feat_dict[chain_ids[0]]["N_atoms"]
        N_atoms_B = dimer_feat_dict[chain_ids[1]]["N_atoms"]
        B = true_coords.size(0)
        N_total = N_atoms_A + N_atoms_B

        chain_mask_A = (torch.arange(N_total, device=true_coords.device) < N_atoms_A).unsqueeze(0).expand(B, -1)
        chain_mask_B = ~chain_mask_A
        combined_mask_A = crop_mask & chain_mask_A
        combined_mask_B = crop_mask & chain_mask_B

        # Chain A:
        self.test_single_chain_rmsd.update(
            pred_coords*self.scale_true_coords,
            true_coords*self.scale_true_coords,
            combined_mask_A.float()
        )
        monomer_chainA_rmsd = self.test_single_chain_rmsd.last_value
        # aligned on A RMSD calculated on B:
        self.test_cross_chain_rmsd_A.update(
            pred_coords*self.scale_true_coords,
            true_coords*self.scale_true_coords,
            combined_mask_A.float()
        )
        cross_chain_rmsd_A = self.test_cross_chain_rmsd_A.last_value
        
        # Chain B:
        self.test_single_chain_rmsd.update(
            pred_coords*self.scale_true_coords,
            true_coords*self.scale_true_coords,
            combined_mask_B.float()
        )
        monomer_chainB_rmsd = self.test_single_chain_rmsd.last_value
        # aligned on B RMSDcalculated on A:
        self.test_cross_chain_rmsd_B.update(
            pred_coords*self.scale_true_coords,
            true_coords*self.scale_true_coords,
            combined_mask_B.float()
        )
        cross_chain_rmsd_B = self.test_cross_chain_rmsd_B.last_value
        
        self.log(
            "test/monomer_rmsd(A)",
            self.test_single_chain_rmsd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        ########

        ######## DockQ:
        model = load_PDB(file_pred)
        native = load_PDB(file_true)

        dockq_out_dict = run_on_all_native_interfaces(model, native)
        dockq = None
        fnat = None
        irmsd = None
        lrmsd = None
        protein_id = list(dimer_feat_dict.values())[0]["protein_id"]
        dockq_failed = 0
        if dockq_out_dict[1] == 0:
            dockq_failed = 1
            # Can fail if interface distance in native structure is too large for AFDDI pseudo-dimers.
        else:
            # Updated metrics only if dockq succeeds.

            dockq = dockq_out_dict[0]['AB']['DockQ']
            fnat = dockq_out_dict[0]['AB']['fnat']
            irmsd = dockq_out_dict[0]['AB']['iRMSD']
            lrmsd = dockq_out_dict[0]['AB']['LRMSD']
            
            self.test_dockq.update(dockq)
            self.test_fnat_dockq.update(fnat)
            self.test_iRMSD_dockq.update(irmsd)
            self.test_LRMSD_dockq.update(lrmsd)

            print(f"dimer_rmsd: {dimer_rmsd}, monomer_chainA_rmsd: {monomer_chainA_rmsd}, monomer_chainB_rmsd: {monomer_chainB_rmsd}, cross_chain_rmsd_A: {cross_chain_rmsd_A}, cross_chain_rmsd_B: {cross_chain_rmsd_B}, DockQ: {dockq}, fnat: {fnat}, irmsd: {irmsd}, lrmsd: {lrmsd}")

            self.log(
                "test/dockq(0 to 1)",
                self.test_dockq,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/fnat_dockq(0 to 1)",
                self.test_fnat_dockq,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/iRMSD_dockq(A)",
                self.test_iRMSD_dockq,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/LRMSD_dockq(A)",
                self.test_LRMSD_dockq,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/cross_chain_rmsd_A",
                self.test_cross_chain_rmsd_A,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "test/cross_chain_rmsd_B",
                self.test_cross_chain_rmsd_B,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Log count of failing dockq evaluations:
        self.test_failing_dockq.update(dockq_failed)
        self.log(
            "test/failing_dockq(count)",
            self.test_failing_dockq,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        ########

        self.test_records.append(
            {
                "protein_id": protein_id,
                "dimer_rmsd": self._to_float(dimer_rmsd),
                "monomer_chainA_rmsd": self._to_float(monomer_chainA_rmsd),
                "monomer_chainB_rmsd": self._to_float(monomer_chainB_rmsd),
                "cross_chain_rmsd_A": self._to_float(cross_chain_rmsd_A),
                "cross_chain_rmsd_B": self._to_float(cross_chain_rmsd_B),
                "dockq": self._to_float(dockq),
                "fnat": self._to_float(fnat),
                "irmsd": self._to_float(irmsd),
                "lrmsd": self._to_float(lrmsd),
                "dockq_failed": dockq_failed,
            }
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
        # dimer_feat_dict = self.sampler.sample_with_plot(dimer_feat_dict, multiplicity, trainer_root_dir=self.trainer.default_root_dir, interpolant=self.interpolant)

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
        return pred_coords, file, file_true


    def on_test_epoch_end(self):
        records = self.test_records
        if dist.is_available() and dist.is_initialized():
            gathered = None
            if dist.get_rank() == 0:
                gathered = [None for _ in range(dist.get_world_size())]
            dist.gather_object(records, gathered, dst=0)
            if dist.get_rank() == 0:
                records = [r for sub in gathered for r in sub]

        if self.trainer.is_global_zero:
            out_path = os.path.join(self.trainer.default_root_dir, "test_metrics.csv")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fieldnames = [
                "protein_id",
                "dimer_rmsd",
                "monomer_chainA_rmsd",
                "monomer_chainB_rmsd",
                "cross_chain_rmsd_A",
                "cross_chain_rmsd_B",
                "dockq",
                "fnat",
                "irmsd",
                "lrmsd",
                "dockq_failed",
            ]
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
