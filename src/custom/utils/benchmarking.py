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

    def update(self, pred_coords: torch.Tensor, true_coords: torch.Tensor):
        atom_mask = true_coords.new_ones(true_coords.shape[:2])
        rmsd_per_batch, _ = compute_aligned_RMSD(pred_coords, true_coords, atom_mask) # (B,)
        self.sum_rmsd += rmsd_per_batch.sum()
        self.n_samples += rmsd_per_batch.numel()

    def compute(self) -> torch.Tensor:
        # mean RMSD over all samples in the epoch (and across GPUs)
        return self.sum_rmsd / torch.clamp(self.n_samples, min=1)


