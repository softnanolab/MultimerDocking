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
    return rmsd


def crop_first_last_M_residues_from_atom_res_idx(res_idx: torch.Tensor, M: int) -> torch.Tensor:
    """
    res_idx: (B, N_atoms, 1) or (B, N_atoms) integer residue indices per atom
    returns atom_mask: (B, N_atoms) bool, True = keep
    """
    if res_idx.dim() == 3:
        res = res_idx.squeeze(-1)
    else:
        res = res_idx
    assert res.dim() == 2, res.shape

    # Handle any padding atoms if you have them: you must define how they look.
    # If you have an existing atom_mask for valid atoms, apply it here.
    # For now assume all atoms valid.
    res_min = res.amin(dim=1, keepdim=True)  # (B,1)
    res_max = res.amax(dim=1, keepdim=True)  # (B,1)

    low  = res_min + M
    high = res_max - M

    keep = (res >= low) & (res <= high)      # (B, N_atoms)

    # Optional: if M is too large, this will create all-False masks; decide policy:
    # keep = keep & (high >= low)  # ensures empty if impossible
    return keep # (B, N_atoms)


def create_crop_mask(dimer_feat_dict: dict, M: int) -> torch.Tensor:
    """
    Returns a boolean mask cropping M residues from the N- and C- terminus of each chain.
    returns mask with shape (B, N_atoms_A + N_atoms_B), True = keep
    """
    chain_ids = list(dimer_feat_dict.keys())
    res_id_A = dimer_feat_dict[chain_ids[0]]["res_id"] # (B, N_atoms_A, 1), specifies the residue id for each atom
    res_id_B = dimer_feat_dict[chain_ids[1]]["res_id"] # (B, N_atoms_B, 1)
    if M is None:
        return torch.ones(res_id_A.shape[0], res_id_A.shape[1] + res_id_B.shape[1], dtype=torch.bool, device=res_id_A.device)
    crop_mask_A = crop_first_last_M_residues_from_atom_res_idx(res_id_A, M=M) # (B, N_atoms_A)
    crop_mask_B = crop_first_last_M_residues_from_atom_res_idx(res_id_B, M=M) # (B, N_atoms_B)
    crop_mask = torch.cat([crop_mask_A, crop_mask_B], dim=1) # (B, N_atoms_A + N_atoms_B)
    return crop_mask # boolean with True = keep, (B, N_atoms_A + N_atoms_B)