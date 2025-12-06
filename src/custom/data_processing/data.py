"""
- This module defines the Dataset and DataLoader.
- Furthermore, the data preprocessing after loading the .pt files and before the models forward pass is defined.
"""


import json

import torch
import torch.nn.functional as F
from tqdm import tqdm

from simplefold.utils.boltz_utils import center_random_rotation


########################################################
####                 Data filtering                #####
########################################################
class Filter:
    """
    Filter class for filtering data.
    """
    def __init__(self, filters: list[callable] = None):
        self.filters = filters if filters is not None else []

    def __call__(self, entry: dict) -> bool:
        if not self.filters:
            return True  # If no filters i.e. filters is None or empty list, include all entries
        results = []
        for filter_fn in self.filters:
            results.append(filter_fn(entry))
        return all(results)

def max_combined_sequence_length(entry: dict, N_max: int) -> bool:
    """
    Filters entries by combined sequence length.
    Args:
        N_max: Maximum allowed combined sequence length
    """
    combined_length = sum(entry["sequence_lengths"])
    if combined_length <= N_max:
        return True
    else:
        return False

def min_individual_sequence_length(entry: dict, N_min: int) -> bool:
    """
    Filters entries by individual sequence length.
    Args:
        N_min: Minimum allowed individual sequence length
    """
    for sequence_length in entry["sequence_lengths"]:
        if sequence_length < N_min:
            return False
    return True

def train_filter(entry: dict) -> bool:
    """
    Filters entries by whether they are in the train set.
    """
    if entry["split"] == "train":
        return True
    else:
        return False

def val_filter(entry: dict) -> bool:
    """
    Filters entries by whether they are in the validation set.
    """
    if entry["split"] == "val":
        return True
    else:
        return False

def test_filter(entry: dict) -> bool:
    """
    Filters entries by whether they are in the test set.
    """
    if entry["split"] == "test":
        return True
    else:
        return False

def filter_data(manifest_path: str, Filter: Filter = None) -> list[str]:
    """
    Parses the manifest.jsonl and filters the data according to the given Filter object.
    Processes one line at a time for memory efficiency.
    Args:
        manifest_path: Path to the manifest.jsonl file
        Filter: Optional Filter object that takes a dict (parsed JSON line) and returns bool.
                If None, all entries are included.
    Returns:
        List of data paths extracted from the entries that pass the Filter criteria.
    """
    filtered_file_list = []
    with open(manifest_path, "r") as f:
        for line in tqdm(f, desc="Filtering manifest"):
            entry = json.loads(line)
            if Filter is not None:
                if not Filter(entry):
                    continue  # Skip this entry if it doesn't pass the filter
            # Include the entry if it passes the filter (or if no filter is provided)
            filtered_file_list.append(entry["path"])
    return filtered_file_list


########################################################
####             Dataset and DataLoader            #####
########################################################
class AFDDI_Dataset(torch.utils.data.Dataset):
    """
    Dataset for AFDDI.
    """
    def __init__(self, manifest_path: str, Filter: Filter = None):
        self.file_list = filter_data(manifest_path, Filter=Filter)

        if len(self.file_list) == 0:
            raise ValueError(f"No files in {manifest_path}.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        sample = torch.load(path, weights_only=False)
        return sample

def AFDDI_collate_fn(batch):
    """
    Collate function for AFDDI dataset.
    """
    return batch


########################################################
####        Data preprocessing after loading       #####
########################################################
def scale_coords(batch, scale_true_coords, scale_ref_coords):
    """
    Scales the true and reference coordinates by the given scale factors.
    """
    for feat_dict in batch:
        for chain_dict in feat_dict.values():
            chain_dict["true_coords"] = chain_dict["true_coords"] / scale_true_coords
            chain_dict["ref_pos"] = chain_dict["ref_pos"] / scale_ref_coords
    return batch

def center_and_rotate_chains(batch, multiplicity: int, device: torch.device):
    """
    Adds augmented coords as "augmented_coords" in the chain dicts which are independently centered and randomly rotated chains coords.
    Inputs:
        -batch: list of feature dictionaries, one dict for each multimer. The feature dictionaries contain the chain dictionaries.
    Outputs:
        - the updated batch with the augmented coords added to the chain dictionaries as chain_dict["augmented_coords"]=tensor (1, N_atoms, 3).
    """
    for feat_dict in batch:
        for chain_dict in feat_dict.values():
            true_coords = chain_dict["true_coords"] # (1, N_atoms, 3)

            augmented_coords = true_coords.repeat_interleave(multiplicity, dim=0).float() # (multiplicity = B, N_atoms, 3)
            B, N_atoms, _ = augmented_coords.shape
            atom_mask = torch.ones(B, N_atoms, device=device) # (B, N_atoms). Augmentation function requires an atom mask, which is currently trivially 1 for all atoms.
            augmented_coords = center_random_rotation(augmented_coords, atom_mask=atom_mask, rotation=True, centering=True, return_second_coords=False, second_coords=None)
            
            chain_dict["augmented_coords"] = augmented_coords
    return batch

def prepare_input_features(dimer_feat_dict, multiplicity: int, device: torch.device):
    """
    Prepares features for the forward pass. This includes feature addition, reshaping and multiplicity handling.
    """
    for chain_id, chain_dict in dimer_feat_dict.items():

        chain_dict["pLM_emb"] = chain_dict["pLM_emb"].repeat_interleave(multiplicity, dim=0).float() # (B, N_r, pLM_num_layers, d_e)

        chain_dict["true_coords"] = chain_dict["true_coords"].repeat_interleave(multiplicity, dim=0).float() # (B, N_atoms, 3)

        N_r = len(chain_dict["sequence"])
        chain_dict["sequence_length"] = torch.tensor([N_r], device=device).repeat_interleave(multiplicity, dim=0).unsqueeze(-1).float() # (B, 1)

        chain_dict["atom_to_token"] = F.one_hot(chain_dict["res_id"].long(), num_classes=N_r).float().repeat_interleave(multiplicity, dim=0) # (B, N_atoms, N_r)

        chain_dict["res_id"] = chain_dict["res_id"].float().repeat_interleave(multiplicity, dim=0).unsqueeze(-1) # (B, N_atoms, 1)

        chain_dict["ref_pos"] = chain_dict["ref_pos"].float().repeat_interleave(multiplicity, dim=0) # (B, N_atoms, 3)

        chain_dict["charge"] = chain_dict["charge"].float().repeat_interleave(multiplicity, dim=0).unsqueeze(-1) # (B, N_atoms, 1)

        chain_dict["atomic_number"] = F.one_hot(chain_dict["atomic_number"].long(), num_classes=128).float().repeat_interleave(multiplicity, dim=0) # (B, N_atoms, 128)

        chain_dict["atom_name_feat"] = F.one_hot(chain_dict["atom_name_feat"].long(), num_classes=64).float().repeat_interleave(multiplicity, dim=0) # (B, N_atoms, 4, 64)
        B, N_atoms, ch, bin = chain_dict["atom_name_feat"].shape
        chain_dict["atom_name_feat"] = chain_dict["atom_name_feat"].reshape(B, N_atoms, ch * bin).float() # (B, N_atoms, 256)

        chain_dict["res_type_feat"] = F.one_hot(chain_dict["res_type_feat"].long(), num_classes=23).float().repeat_interleave(multiplicity, dim=0) # (B, N_atoms, 23)
        
    return dimer_feat_dict

def merge_chains(dimer_feat_dict, key: str):
    """
    Concatenates the atom feature specified by key of the chain_dicts together along the atom_dim in the order the chains appear in the dimer_feat_dict.
    Assumes shape (B, N_atoms, *).
    Returns the concatenated tensor.
    """
    concatenated_tensor = torch.cat([chain_dict[key] for chain_dict in dimer_feat_dict.values()], dim=1) # (B, N_atoms_A + N_atoms_B, *)
    return concatenated_tensor

def split_chains(dimer_feat_dict, key, concatenated_tensor):
    """
    Splits the concatenated tensor back into the individual chain dicts, according to chain order and number of atoms per chain.
    Assumes concatenated_tensor shape (B, N_atoms_A + N_atoms_B, *).
    The slicing is done along the atom dim.
    """
    offset = 0
    for chain_dict in dimer_feat_dict.values():
        N_atoms = chain_dict["true_coords"].shape[1]
        chain_dict[key] = concatenated_tensor[:, offset:offset+N_atoms, ...]
        offset += N_atoms
    return dimer_feat_dict

def extract_full_dimer_coords(dimer_feat_dict, device: torch.device):
    """
    Build the full dimer coords from the chains, centers and randomly rotates the full dimer and returns it.
    Note that the chains are stacked concatenated along the atom dim according to their order in the dimer_feat_dict.
    """
    full_dimer_coords = merge_chains(dimer_feat_dict, "true_coords") # (B, N_atoms_A + N_atoms_B, 3)
    B, N_atoms, _ = full_dimer_coords.shape
    atom_mask = torch.ones(B, N_atoms, device=device) # (B, N_atoms). Augmentation function requires an atom mask, which is currently trivially 1 for all atoms.
    full_dimer_coords = center_random_rotation(full_dimer_coords, atom_mask=atom_mask, rotation=True, centering=True, return_second_coords=False, second_coords=None)
    return full_dimer_coords

