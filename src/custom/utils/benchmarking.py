"""
Contains utilities for benchmarking the model on the test set.
"""


from torchmetrics import Metric
import torch

from custom.utils.structure_utils import compute_aligned_RMSD


class MeanRMSD(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total RMSD and total number of samples
        self.add_state("sum_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.last_value = None

    def update(self, pred_coords: torch.Tensor, true_coords: torch.Tensor, atom_mask: torch.Tensor):
        rmsd_per_batch, _ = compute_aligned_RMSD(pred_coords, true_coords, atom_mask) # (B,)
        sum = rmsd_per_batch.sum()
        self.sum_rmsd += sum
        self.last_value = rmsd_per_batch.mean()
        self.n_samples += rmsd_per_batch.numel()


    def compute(self) -> torch.Tensor:
        # mean RMSD over all samples in the epoch (and across GPUs)
        return self.sum_rmsd / torch.clamp(self.n_samples, min=1)


class SingleChainRMSD(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total RMSD and total number of samples
        self.add_state("sum_rmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.last_value = None

    def update(self, pred_coords: torch.Tensor, true_coords: torch.Tensor, atom_mask: torch.Tensor):
        """
        Atom mask must specify the chain atoms to consider for the RMSD calculation with 1.0 and 0.0 for atoms to ignore.
        """
        rmsd_per_batch, _ = compute_aligned_RMSD(pred_coords, true_coords, atom_mask) # (B,)
        sum = rmsd_per_batch.sum()
        self.sum_rmsd += sum
        self.last_value = rmsd_per_batch.mean()
        self.n_samples += rmsd_per_batch.numel()


    def compute(self) -> torch.Tensor:
        # mean RMSD over all samples in the epoch (and across GPUs)
        return self.sum_rmsd / torch.clamp(self.n_samples, min=1)


class DockQMetric(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total DockQ and total number of samples
        self.add_state("sum_dockq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, dockq: float):
        """
        dockq_out_dict is a dictionary containing the DockQ scores for each native interface.
        """

        self.sum_dockq += dockq
        self.n_samples += 1

    def compute(self) -> torch.Tensor:
        return self.sum_dockq / torch.clamp(self.n_samples, min=1)


class fnat_DockQMetric(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total DockQ and total number of samples
        self.add_state("sum_fnat", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, fnat: float):
        """
        dockq_out_dict is a dictionary containing the DockQ scores for each native interface.
        """

        self.sum_fnat += fnat
        self.n_samples += 1

    def compute(self) -> torch.Tensor:
        return self.sum_fnat / torch.clamp(self.n_samples, min=1)


class iRMSD_DockQMetric(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total DockQ and total number of samples
        self.add_state("sum_irmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, irmsd: float):
        """
        dockq_out_dict is a dictionary containing the DockQ scores for each native interface.
        """
        self.sum_irmsd += irmsd
        self.n_samples += 1

    def compute(self) -> torch.Tensor:
        return self.sum_irmsd / torch.clamp(self.n_samples, min=1)


class LRMSD_DockQMetric(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total DockQ and total number of samples
        self.add_state("sum_lrmsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, lrmsd: float):
        """
        dockq_out_dict is a dictionary containing the DockQ scores for each native interface.
        """
        self.sum_lrmsd += lrmsd
        self.n_samples += 1
    
    def compute(self) -> torch.Tensor:
        return self.sum_lrmsd / torch.clamp(self.n_samples, min=1)


class FailingDockQMetric(Metric):
    def __init__(self):
        super().__init__()
        # accumulate total DockQ and total number of samples
        self.add_state("sum_dockq_failed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, dockq_failed: int):
        """
        Increments when dockq fails to evaluate.
        dock_failed is 1 if dockq failed to evaluate, 0 otherwise.
        """
        self.n_samples += dockq_failed
    
    def compute(self) -> torch.Tensor:
        return self.n_samples