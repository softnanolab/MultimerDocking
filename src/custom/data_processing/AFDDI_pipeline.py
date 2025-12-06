'''Process the AFDB DDI from raw pdb files into .pt files 
containing coordinates and relevant features.'''

import torch
import numpy as np
import pathlib as pl
import re
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import json

import biotite.structure as struc
import biotite.structure.info as struc_info

from atomworks.ml.transforms.base import Transform, Compose
from atomworks.io import parse as atomworks_parse


########################################################
####         Data import and preprocessing         #####
########################################################
def group_exact_pairs(dataset_path: str) -> list[tuple[str, str]]:
    '''Group the AFDB DDI file paths from the given directory into pairs of dimers. Drop ids with != 2 files(chains).
    Pairs are returned as a list oftuples of the file paths.'''
    dataset_path = pl.Path(dataset_path)
    all_files = list(dataset_path.glob("*.pdb"))
    N_files_old = len(all_files)
    print(f"Found {N_files_old} files(chains) in the dataset.")
    groups = defaultdict(list)
    for f in all_files:
        groups[re.match(r"(.*)_D\d+", f.stem).group(1)].append(str(f))
    dimer_files = [tuple(v) for v in groups.values() if len(v) == 2]
    print(f"Filtered to {len(dimer_files)} dimer pairs. Dropped {(N_files_old - len(dimer_files)*2)/N_files_old*100:.1f}% of the chains because num_chains != 2.")
    return dimer_files


########################################################
####            Atomworks transforms               #####
########################################################
class TestInputData(Transform):
    """Verifies the input data is valid."""

    requires_previous_transforms = []
    incompatible_previous_transforms = []

    def __init__(self, annotation_name: str = "ref_pos") -> None:
        self.annotation_name = annotation_name

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        assert (atom_array.is_polymer == True).all()
        assert (atom_array.chain_type == 6).all() # chain type 6 is polypeptide_L, see: https://github.com/RosettaCommons/atomworks/blob/production/src/atomworks/enums.py
        num_chains = len(np.unique(atom_array.chain_id))
        assert num_chains == 2 # Make sure there are exactly two chains with unique ids
        return data

class DropOXTAtoms(Transform):
    """Drop OXT atoms."""

    requires_previous_transforms = [TestInputData,]
    incompatible_previous_transforms = []

    def __init__(self) -> None:
        pass

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        mask = (atom_array.atom_name != "OXT")
        atom_array = atom_array[mask]

        data["atom_array"] = atom_array
        return data

class BackboneOnly(Transform):
    """Drop all atoms except for the backbone."""

    requires_previous_transforms = [TestInputData,]
    incompatible_previous_transforms = []

    def __init__(self, backbone_only: bool) -> None:
        self.backbone_only = backbone_only

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        if self.backbone_only:
            mask = np.isin(atom_array.atom_name, ["N", "CA", "C", "O"])
            atom_array = atom_array[mask]
        data["atom_array"] = atom_array
        return data

class AnnotateResidueId(Transform):
    """Annotate each atom with a monotonically increasing residue id. Independent per chain controls the starting id for each chain."""

    requires_previous_transforms = [TestInputData,]
    incompatible_previous_transforms = []

    def __init__(self, annotation_name: str = "res_id", independent_per_chain: bool = True) -> None:
        self.annotation_name = annotation_name
        self.independent_per_chain = independent_per_chain

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        residue_ids = np.empty(atom_array.array_length(), dtype=np.int32)

        if self.independent_per_chain:
            chain_ids = atom_array.chain_id
            for chain in np.unique(chain_ids):
                chain_mask = (chain_ids == chain)
                local_ids = struc.get_all_residue_positions(atom_array[chain_mask])
                residue_ids[chain_mask] = local_ids.astype(np.int32)
        else:
            residue_ids = struc.get_all_residue_positions(atom_array).astype(np.int32)

        atom_array.set_annotation(self.annotation_name, residue_ids)
        data["atom_array"] = atom_array
        return data


class AnnotateOneLetterCode(Transform):
    """Annotate each atom with a one-letter residue code."""

    requires_previous_transforms = [TestInputData,]
    incompatible_previous_transforms = []

    def __init__(self, annotation_name: str = "one_letter_code") -> None:
        self.annotation_name = annotation_name

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]

        one_letter = []
        for res_name in atom_array.res_name:
            code = struc_info.one_letter_code(res_name)
            if code is None:
                code = "X" # fallback
            one_letter.append(code)

        one_letter = np.array(one_letter, dtype="U1")
        atom_array.set_annotation(self.annotation_name, one_letter)

        data["atom_array"] = atom_array
        return data


class AnnotateResTypeFeat(Transform):
    """Annotate each atom with residue type as integer index in token dictionary.
    
    Token dictionary: <pad>, -, A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V, X
    Maps from one-letter codes to integer indices (NOT one-hot encoded).
    """
    
    requires_previous_transforms = [TestInputData, AnnotateOneLetterCode,]
    incompatible_previous_transforms = []
    
    def __init__(self, annotation_name: str = "res_type_feat") -> None:
        self.annotation_name = annotation_name
        # Token dictionary: <pad>, -, 20 amino acids, X (unknown)
        self.tokens = [
            "<pad>",
            "-",
            "A",  # Alanine
            "R",  # Arginine
            "N",  # Asparagine
            "D",  # Aspartic acid
            "C",  # Cysteine
            "Q",  # Glutamine
            "E",  # Glutamic acid
            "G",  # Glycine
            "H",  # Histidine
            "I",  # Isoleucine
            "L",  # Leucine
            "K",  # Lysine
            "M",  # Methionine
            "F",  # Phenylalanine
            "P",  # Proline
            "S",  # Serine
            "T",  # Threonine
            "W",  # Tryptophan
            "Y",  # Tyrosine
            "V",  # Valine
            "X",  # Unknown
        ]
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        self.num_tokens = len(self.tokens)
    
    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)
    
    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        
        # Map one-letter codes to token IDs
        res_type_feat = []
        for code in atom_array.one_letter_code:
            # Map to token ID, default to "X" (unknown) if not found:
            token_id = self.token_to_id.get(str(code).upper(), self.token_to_id["X"])
            res_type_feat.append(token_id)
        
        res_type_feat = np.array(res_type_feat, dtype=np.int32)
        atom_array.set_annotation(self.annotation_name, res_type_feat)
        
        data["atom_array"] = atom_array
        return data


class AnnotateAtomNameFeat(Transform):
    """Annotate each atom with its pdb atom name (N,C,CA,CB,CD,CD1,CD2,...) converted to a 4 integers feature vector."""
    
    requires_previous_transforms = [TestInputData,]
    incompatible_previous_transforms = []
    
    def __init__(self, annotation_name: str = "atom_name_feat") -> None:
        self.annotation_name = annotation_name
    
    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)
    
    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        
        def convert_atom_name(name: str) -> tuple[int, int, int, int]:
            """Convert an atom name to a standard format (4 integers)."""
            name = name.strip()
            name = [ord(c) - 32 for c in name]
            name = name + [0] * (4 - len(name))
            return tuple(name)
        
        atom_name_feat = []
        for atom_name in atom_array.atom_name:
            converted = convert_atom_name(atom_name)
            atom_name_feat.append(converted)
        
        # Convert to numpy array of shape (N_atoms, 4) with dtype int32
        atom_name_feat = np.array(atom_name_feat, dtype=np.int32)
        atom_array.set_annotation(self.annotation_name, atom_name_feat)
        
        data["atom_array"] = atom_array
        return data


aa_1_to_3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
       "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
       "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
       "U":"SEC","O":"PYL"}


def annotate_ref_pos(atom_array, annotation_name):
    """Annotate each atom with CCD ideal residue coordinates (CCD frame, i.e. local residue frames)."""
    atom_array = atom_array.copy() # Atomworks convention is to not modify inplace with the function.
    n = atom_array.array_length()
    ideal = np.full((n, 3), np.nan, float)

    starts = struc.get_residue_starts(atom_array)
    ends = np.append(starts[1:], n)

    for s, e in zip(starts, ends):
        try:
            res = atom_array[s:e] # get residue atoms
            res_name = aa_1_to_3[res.one_letter_code[0]] # map to standard residue name
            tmpl = struc_info.residue(res_name) 
            tmpl = tmpl[tmpl.element != "H"] # remove hydrogens
            name2coord = {n: c for n, c in zip(tmpl.atom_name, tmpl.coord)}

            for i, name in enumerate(res.atom_name):
                ref_coord = name2coord.get(name)
                ideal[s + i] = ref_coord
        except:
            continue # if it fails the coordinates are set to nan
    
    atom_array.set_annotation(annotation_name, ideal)
    return atom_array


class AnnotateRefPos(Transform):
    """Annotate each atom with CCD ideal residue coordinates (CCD frame, i.e. local residue frames)."""

    requires_previous_transforms = [TestInputData,AnnotateOneLetterCode,]
    incompatible_previous_transforms = []

    def __init__(self, annotation_name: str = "ref_pos") -> None:
        self.annotation_name = annotation_name

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        atom_array = data["atom_array"]
        atom_array = annotate_ref_pos(atom_array, self.annotation_name)
        data["atom_array"] = atom_array
        return data


def build_feature_dict(data: dict, feat_dict_key_name: str) -> dict:
    """Build the feature dictionary for the PDB Multimers dataset.
       The feature dictionary contains a dictionary for each protein chain,
       which contains the features for each chain."""
    
    atom_array = data["atom_array"]
    chains = np.unique(atom_array.chain_id)
    assert len(chains) == 2, "Expected 2 chains (dimer), got %d" % len(chains)
    
    feat_dict = {}
    for chain_id in chains:
        chain_dict = {}
        chain_mask = (atom_array.chain_id == chain_id)
        chain_array = atom_array[chain_mask]


        # One letter AA sequence string:
        res_idx = struc.get_residue_starts(chain_array)
        sequence = chain_array.one_letter_code[res_idx]
        sequence = "".join(sequence)
        chain_dict["sequence"] = sequence
        
        # Atom coordinates:
        chain_dict["true_coords"] = torch.from_numpy(chain_array.coord).unsqueeze(0)

        # Atom sequence position:
        chain_dict["res_id"] = torch.from_numpy(chain_array.res_id).unsqueeze(0)

        # Reference conformer coordinates:
        chain_dict["ref_pos"] = torch.from_numpy(chain_array.ref_pos).unsqueeze(0)

        # Residue type:
        chain_dict["res_type_feat"] = torch.from_numpy(chain_array.res_type_feat).unsqueeze(0)
        
        # Charge:
        chain_dict["charge"] = torch.from_numpy(chain_array.charge).unsqueeze(0)
        
        # Atomic number:
        chain_dict["atomic_number"] = torch.from_numpy(chain_array.atomic_number).unsqueeze(0) # (1, N_atoms)
        
        # Canonical PDB atom name:
        chain_dict["atom_name"] = chain_array.atom_name
        chain_dict["atom_name_feat"] = torch.from_numpy(chain_array.atom_name_feat).unsqueeze(0)

        # Add chain to the feature dictionary:
        feat_dict[str(chain_id)] = chain_dict
    
    data[feat_dict_key_name] = feat_dict
    return data

class BuildFeatureDict(Transform):
    """Builds feature dictionary for the dimer."""

    requires_previous_transforms = [
        TestInputData,
        DropOXTAtoms,
        BackboneOnly,
        AnnotateResidueId,
        AnnotateOneLetterCode,
        AnnotateResTypeFeat,
        AnnotateAtomNameFeat,
        AnnotateRefPos,
    ]
    incompatible_previous_transforms = []

    def __init__(self, data_key_name: str = "feat_dict") -> None:
        self.data_key_name = data_key_name

    def check_input(self, data: dict) -> None:
        assert "atom_array" in data
        assert isinstance(data["atom_array"], struc.AtomArray)

    def forward(self, data: dict) -> dict:
        data = build_feature_dict(data, self.data_key_name)
        return data


########################################################
####       Pipeline and final processing           #####
########################################################
def build_pipeline(backbone_only: bool) -> Compose:
    """Build the pipeline for processing the PDB Multimers dataset as prepared for mint (already contains dimers only)."""
    pipeline = Compose([
        TestInputData(),
        DropOXTAtoms(),
        BackboneOnly(backbone_only),
        AnnotateResidueId(),
        AnnotateOneLetterCode(),
        AnnotateResTypeFeat(),
        AnnotateAtomNameFeat(),
        AnnotateRefPos(),
        BuildFeatureDict(),
    ])
    return pipeline

def save_feature_dict(feat_dict: dict, filepath: str) -> None:
    """Save the feature dictionary to a torch .pt file."""
    torch.save(feat_dict, filepath)
    return None

def add_to_manifest(manifest_path: str, json_dict: dict) -> None:
    """Add the path string as a new row to the manifest.jsonl."""
    with open(manifest_path, "a") as f:
        f.write(json.dumps(json_dict) + "\n")
    return None

def process_dimer(args: tuple[tuple[str, str], str, bool]) -> dict:
    """Extract the feature dictionary for a multimer in cif_file and save it to a torch .pt file at output_path."""
    chain_files = args[0]
    output_path = args[1]
    backbone_only = args[2]

    chain_files = [pl.Path(chain_file) for chain_file in chain_files]
    output_path = pl.Path(output_path)
    files_path = output_path / "pt_files"
    files_path.mkdir(parents=True, exist_ok=True)

    # pdb import and some cleaning with atomworks parse:
    parsed_chains = [atomworks_parse(chain_file, hydrogen_policy="remove") for chain_file in chain_files]
    atom_arrays = [parsed_chain["asym_unit"][0] for parsed_chain in parsed_chains]

    assert len(atom_arrays) == 2, "Expected 2 chains (dimer), got %d" % len(atom_arrays)

    # Set the chain ids to the file chain ids:
    chain_ids = [chain_file.stem.split("_")[1] for chain_file in chain_files]
    for chain_id, atom_array in zip(chain_ids, atom_arrays):
        atom_array.chain_id = np.full(len(atom_array), chain_id) # list gets modified in-place
    
    # Concatenate the chains into a single atom_array for the atomworks pipeline:
    joint_atom_array = struc.concatenate(atom_arrays)

    # Feed through atomworks transforms pipeline:
    in_data = {"atom_array": joint_atom_array}
    pipeline = build_pipeline(backbone_only)
    out_data = pipeline(in_data)

    feature_dict = out_data["feat_dict"]

    # Adds protein id to each chain dict in the feature dict.
    # Note that adding data except chain dicts to the feature dict is not supported and will lead to errors:
    protein_id = chain_files[0].stem.split('_')[0]
    for _, chain_dict in feature_dict.items():
        chain_dict["protein_id"] = protein_id

    out_file = files_path / f"{protein_id}.pt"
    save_feature_dict(feature_dict, out_file)

    split = np.random.choice(["train", "val", "test"], p=[0.8, 0.1, 0.1])

    out_json = {
        "path": str(out_file), 
        "chain_ids": [str(chain_id) for chain_id in chain_ids],
        "sequence_lengths": [len(feature_dict[chain_id]["sequence"]) for chain_id in chain_ids],
        "backbone_only": backbone_only,
        "split": split,
    }
    return out_json

def process_dataset(
    dataset_path: str,
    output_path: str,
    N_workers: int = 1,
    backbone_only: bool = True,
    ) -> None:
    """Extracts the feature dict for each dimer from all pdb files in dataset_path dir and saves dimer feats. to output_path dir as .pt files."""
    # Convert N_workers to int in case it's passed as a string from command line:
    N_workers = int(N_workers)
    dataset_path = pl.Path(dataset_path)
    output_path = pl.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True) # Create the output directory if it doesn't exist.

    # Check if manifest.jsonl exists and if yes delete the old one:
    manifest_path = output_path / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    dimer_files = group_exact_pairs(dataset_path)
    args = [(dimer_pair, output_path, backbone_only) for dimer_pair in dimer_files]

    print(f"Starting to process {len(args)} dimers with {N_workers} workers...")
    if backbone_only:
        print("Extracting backbone only...")
    else:
        print("Extracting full heavy atoms set...")

    with Pool(processes=N_workers) as pool:
        for out_json in tqdm(pool.imap_unordered(process_dimer, args),
                             desc="Processing AFDB_DDI dataset",
                             total=len(args)):
            add_to_manifest(manifest_path, out_json)
    
    print(f"Saved {len(dimer_files)} dimer .pt files to {output_path}.")
    return None