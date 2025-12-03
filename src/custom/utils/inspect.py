"""
Contains utility functions for inspecting the model, data, and training process.
"""


import pathlib

import torch
import numpy as np
import biotite.structure as struc
from biotite.structure.info import residue as get_residue
from biotite.structure.io import save_structure


aa_1_to_3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
       "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
       "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
       "U":"SEC","O":"PYL"}

def cif_from_tensor(chain_coords: list[torch.Tensor],
                    chain_ids: list[str],
                    seqs: list[str],
                    file: str,
                    backbone_only: bool = False,
                    scale: float = 16.0):
    """
    Saves the chain coords from the chain_coords list to one CIF file.
    IMPORTANT: This function assumes the atoms are listed in CCD PDB canonical order (N,CA,C,O,CB,...). 
    And the residues are listed in the same order as the sequence.
    Hydrogens are not supported.
    Inputs:
    - chain_coords: list[torch.Tensor], each tensor is (1, N_atoms, 3). N_atoms can vary between chains.
    - chain_ids: list[str] of chain_ids
    - seqs: list[str] of one-letter amino acid sequences, one string per chain
    - file: str, the path to the output CIF file
    - backbone_only: bool, set to True if the tensors contain only the backbone atoms (N,CA,C,O)
    - scale: float, coordinates are multiplied by this factor.
    """

    N_total = sum(coords.shape[1] for coords in chain_coords)
    atom_array = struc.AtomArray(N_total)

    offset = 0
    # Iterate over chains:
    for coords, chain_id, seq in zip(chain_coords, chain_ids, seqs):
        assert coords.ndim == 3 and coords.shape[0] == 1 and coords.shape[2] == 3, "ERROR: Coordinates shape mismatch"
        N_atoms = coords.shape[1]
        xyz = coords.detach().cpu().numpy().squeeze(0) # (N_atoms, 3)
        xyz *= scale

        # Construct residue level quantities (depending on number of atoms in the residue):
        seq_3 = [aa_1_to_3[aa] for aa in seq]
        res_id = []
        res_name = []
        atom_name = []
        element = []
        i = 0
        for res_code in seq_3:
            res = get_residue(res_code, allow_missing_coord=False)

            # base mask: drop hydrogens and OXT
            mask = (res.element != "H") & (res.atom_name != "OXT")
            if backbone_only:
                # keep only backbone atoms
                bb_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
                mask &= bb_mask
            res = res[mask]

            res_id.append(res.res_id + i)
            res_name.append(res.res_name)
            atom_name.append(res.atom_name)
            element.append(res.element)
            i += 1
        res_id = np.concatenate(res_id)
        res_name = np.concatenate(res_name)
        atom_name = np.concatenate(atom_name)
        element = np.concatenate(element)

        assert len(res_id) == len(res_name) == len(atom_name) == len(element) == N_atoms, "ERROR: Number of atoms mismatch"

        # Assign chain level quantities:
        atom_array.coord[offset:offset+N_atoms] = xyz
        atom_array.chain_id[offset:offset+N_atoms] = np.full(N_atoms, chain_id)
        atom_array.ins_code[offset:offset+N_atoms] = np.full(N_atoms, "")
        atom_array.hetero[offset:offset+N_atoms] = np.full(N_atoms, False)
        atom_array.add_annotation("occupancy", dtype=np.float32)
        atom_array.occupancy[offset:offset+N_atoms] = np.full(N_atoms, 1.0)
        atom_array.add_annotation("b_factor", dtype=np.float32)
        atom_array.b_factor[offset:offset+N_atoms] = np.full(N_atoms, 0.0)

        # Assign residue level quantities:
        atom_array.res_id[offset:offset+N_atoms] = res_id
        atom_array.res_name[offset:offset+N_atoms] = res_name
        atom_array.atom_name[offset:offset+N_atoms] = atom_name
        atom_array.element[offset:offset+N_atoms] = element
        
        offset += N_atoms

    file = pathlib.Path(file).resolve()
    file.parent.mkdir(parents=True, exist_ok=True)
    save_path = str(file.with_suffix(".cif"))
    save_structure(save_path, atom_array)
    print(f"Saved {save_path}")
    return None