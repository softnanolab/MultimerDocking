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


def aligned_weighted_cross_RMSD(
    pred_atom_coords,
    atom_coords,
    atom_mask,
):
    """
    Takes in a predicted dimer and a ground truth dimer and an atom mask.
    Aligns the ground truth dimer to the predicted dimer but only considering the atoms specified in the atom mask.
    Then the RMSD is computed between predicted dimer and ground truth but only considering the atoms specified in the INVERTED atom mask.
    This RMSD is weighted by the number of atoms in the chain and returned.
    -> I.e.: Align on chains A and compute the RMSD on chains B.

    Inputs:
    - pred_atom_coords: (B, N_atoms, 3)
    - atom_coords: (B, N_atoms, 3)
    - atom_mask: (B, N_atoms). 1 for atoms to align on, 0 for atoms to calculate the RMSD on.

    Outputs:
    - rmsd: (B,)
    """
    align_weights = atom_coords.new_ones(atom_coords.shape[:2]) # (B, N_atoms)

    # Aligns the true coords to the predicted coords:
    with torch.no_grad():
        aligned_true_coords = weighted_rigid_align(
            atom_coords, pred_atom_coords, align_weights, mask=atom_mask
        )

    # weighted MSE loss of denoised atom positions
    squared_norm = ((pred_atom_coords - aligned_true_coords) ** 2).sum(dim=-1)

    inverse_mask = 1 - atom_mask

    rmsd = torch.sqrt(
        torch.sum(squared_norm * align_weights * inverse_mask, dim=-1)
        / torch.sum(align_weights * inverse_mask, dim=-1) # normalize by number of atoms
    )
    # rmsd: (B,)
    # atom_coords_aligned_ground_truth: (B, N_atoms, 3)
    return rmsd