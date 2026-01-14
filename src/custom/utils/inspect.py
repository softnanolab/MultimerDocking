"""
Contains utility functions for inspecting the model, data, and training process.
"""


import pathlib
import string

import torch
import numpy as np
import biotite.structure as struc
from biotite.structure.info import residue as get_residue
from biotite.structure.io import save_structure

from Bio.PDB import MMCIFParser, PDBIO
import os
from glob import glob
from natsort import natsorted
from io import StringIO


aa_1_to_3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
       "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
       "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
       "U":"SEC","O":"PYL"}


def _append_polymer_tables(cif_path: pathlib.Path, chain_ids: list[str], seqs: list[str]):
    lines = ["\n#\n"]

    lines.append("loop_\n_entity.id\n_entity.type\n")
    for i in range(1, len(chain_ids) + 1):
        lines.append(f"{i} polymer\n")

    lines.append("loop_\n_entity_poly.entity_id\n_entity_poly.type\n_entity_poly.pdbx_strand_id\n")
    for i, cid in enumerate(chain_ids, start=1):
        lines.append(f"{i} 'polypeptide(L)' '{cid}'\n")

    lines.append("loop_\n_struct_asym.id\n_struct_asym.entity_id\n")
    for i, cid in enumerate(chain_ids, start=1):
        lines.append(f"'{cid}' {i}\n")

    lines.append("loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n_entity_poly_seq.mon_id\n_entity_poly_seq.hetero\n")
    for i, seq in enumerate(seqs, start=1):
        for j, aa in enumerate(seq, start=1):
            lines.append(f"{i} {j} {aa_1_to_3[aa]} n\n")

    with open(cif_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


class AlphabetCounter:
    def __init__(self):
        self._letters = string.ascii_uppercase
        self._idx = 0

    def __call__(self):
        if self._idx >= len(self._letters):
            raise StopIteration("Alphabet exhausted (A–Z).")
        letter = self._letters[self._idx]
        self._idx += 1
        return letter

    def reset(self):
        self._idx = 0


def cif_from_tensor(chain_coords: list[torch.Tensor],
                    chain_ids: list[str],
                    seqs: list[str],
                    file: str,
                    backbone_only: bool = False,
                    scale: float = 16.0,
                    verbose: bool = True):
    """
    Saves the chain coords from the chain_coords list to one CIF file.
    IMPORTANT: This function assumes the atoms are listed in CCD PDB canonical order (N,CA,C,O,CB,...). 
    And the residues are listed in the same order as the sequence.
    Hydrogens are not supported.
    Inputs:
    - chain_coords: list[torch.Tensor], each tensor is (1, N_atoms, 3). N_atoms can vary between chains.
    - chain_ids: list[str] of chain_ids
    - seqs: list[str] of one-letter amino acid sequences of the chains, one seq string per chain
    - file: str, the path to the output CIF file
    - backbone_only: bool, set to True if the tensors contain only the backbone atoms (N,CA,C,O)
    - scale: float, coordinates are multiplied by this factor.
    """

    N_total = sum(coords.shape[1] for coords in chain_coords)
    atom_array = struc.AtomArray(N_total)
    
    # Add annotations once before processing chains
    atom_array.add_annotation("occupancy", dtype=np.float32)
    atom_array.add_annotation("b_factor", dtype=np.float32)

    offset = 0
    # Iterate over chains:
    alphabet = AlphabetCounter()
    alphas = []
    for coords, _, seq in zip(chain_coords, chain_ids, seqs):
        assert coords.ndim == 3 and coords.shape[0] == 1 and coords.shape[2] == 3, "ERROR: Coordinates shape mismatch"
        N_atoms = coords.shape[1]
        xyz = coords.detach().cpu().float().numpy().squeeze(0) # (N_atoms, 3)
        xyz *= scale

        # Construct residue level quantities (depending on number of atoms in the residue):
        seq_3 = []
        for aa in seq:
            if aa not in aa_1_to_3:
                raise ValueError(f"Unknown amino acid '{aa}' in sequence. Supported: {list(aa_1_to_3.keys())}")
            seq_3.append(aa_1_to_3[aa])
        
        res_id = []
        res_name = []
        atom_name = []
        element = []
        i = 1  # Start residue IDs at 1 (PDB convention)
        for res_code in seq_3:
            res = get_residue(res_code, allow_missing_coord=False)

            # base mask: drop hydrogens and OXT
            mask = (res.element != "H") & (res.atom_name != "OXT")
            if backbone_only:
                # keep only backbone atoms
                bb_mask = np.isin(res.atom_name, ["N", "CA", "C", "O"])
                mask &= bb_mask
            res = res[mask]

            # res.res_id from get_residue is typically all 1s, so we set it to i
            res_id.append(np.full(len(res), i))
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
        alpha_chain_id = alphabet()
        alphas.append(alpha_chain_id)
        atom_array.chain_id[offset:offset+N_atoms] = np.full(N_atoms, alpha_chain_id)
        atom_array.ins_code[offset:offset+N_atoms] = np.full(N_atoms, "")
        atom_array.hetero[offset:offset+N_atoms] = np.full(N_atoms, False)
        atom_array.occupancy[offset:offset+N_atoms] = np.full(N_atoms, 1.0)
        atom_array.b_factor[offset:offset+N_atoms] = np.full(N_atoms, 0.0)

        # Assign residue level quantities:
        atom_array.res_id[offset:offset+N_atoms] = res_id
        atom_array.res_name[offset:offset+N_atoms] = res_name
        atom_array.atom_name[offset:offset+N_atoms] = atom_name
        atom_array.element[offset:offset+N_atoms] = element
        
        offset += N_atoms


    file = pathlib.Path(file).resolve()
    file.parent.mkdir(parents=True, exist_ok=True)
    save_path = file.with_suffix(".cif")

    save_structure(str(save_path), atom_array)
    _append_polymer_tables(save_path, alphas, seqs)
    if verbose: print("Saved", save_path)

    return None


def merge_cifs_into_trajectory(cifs_dir: str, output_traj: str):
    """
    - cifs_dir: str, the directory containing the CIF files
    - output_traj: str, the path to the output PDB file. Must end with .pdb.
    """
    parser = MMCIFParser(QUIET=False)
    io = PDBIO()
    cif_files = glob(os.path.join(cifs_dir, "*.cif")) # load all cif paths
    cif_files = natsorted(cif_files) # sort them according to traj number

    # Get chain ID mapping from first file
    first_structure = parser.get_structure("temp", cif_files[0])
    chain_ids = [chain.id for chain in first_structure.get_chains()]
    chain_map = {cid: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[i] 
                 for i, cid in enumerate(chain_ids)}

    with open(output_traj, "w") as out:
        for i, cif_path in enumerate(cif_files, 1):
            structure = parser.get_structure(f"frame{i}", cif_path)
            # Rename chains to single-char for PDB compatibility
            for chain in structure.get_chains():
                if chain.id in chain_map:
                    chain.id = chain_map[chain.id]
            out.write(f"MODEL     {i}\n")
            buf = StringIO()
            io.set_structure(structure)
            io.save(buf)
            lines = buf.getvalue().splitlines()
            if lines and lines[-1].strip() == "END":
                lines = lines[:-1]
            out.write("\n".join(lines) + "\nENDMDL\n")
        out.write("END\n")