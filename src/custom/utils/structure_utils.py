"""
Contains utilities for molecular structure operations on tensors.
"""

import torch

from simplefold.utils.boltz_utils import weighted_rigid_align


def compute_aligned_RMSD(
    pred_atom_coords,
    atom_coords,
    atom_mask,
):
    """Compute rmsd of the aligned atom coordinates.

    Parameters
    ----------
    pred_atom_coords : torch.Tensor
        Predicted atom coordinates
    atom_coords: torch.Tensor
        Ground truth atom coordinates
    atom_mask : torch.Tensor
        Resolved atom mask
    atom_to_token : torch.Tensor
        Atom to token mapping
    mol_type : torch.Tensor
        Atom type

    Returns
    -------
    Tensor
        The rmsd
    Tensor
        The aligned coordinates
    Tensor
        The aligned weights

    """
    align_weights = atom_coords.new_ones(atom_coords.shape[:2])

    # Aligns the true coords to the predicted coords:
    with torch.no_grad():
        atom_coords_aligned_ground_truth = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    mse_loss = ((pred_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(dim=-1)
    rmsd = torch.sqrt(
        torch.sum(mse_loss * align_weights * atom_mask, dim=-1)
        / torch.sum(align_weights * atom_mask, dim=-1)
    )
    # rmsd: (B,)
    # atom_coords_aligned_ground_truth: (B, N_atoms, 3)
    return rmsd, atom_coords_aligned_ground_truth
    