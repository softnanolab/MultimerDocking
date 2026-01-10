"""
Defines function for the protein language model.
"""

import torch
from functools import partial
import numpy as np

# Mint imports:
from mint.model.esm import ESM2
from mint.dataclasses import Sample
from mint.data.collate_fn import CollateFn

########################################################
####                     ESM_2                     #####
########################################################
load_fn = torch.hub.load
esm_registry = {
    "esm2_8M": partial(load_fn, "facebookresearch/esm:main", "esm2_t6_8M_UR50D"),
    "esm2_35M": partial(load_fn, "facebookresearch/esm:main", "esm2_t12_35M_UR50D"),
    "esm2_150M": partial(load_fn, "facebookresearch/esm:main", "esm2_t30_150M_UR50D"),
    "esm2_650M": partial(load_fn, "facebookresearch/esm:main", "esm2_t33_650M_UR50D"),
    "esm2_3B": partial(load_fn, "facebookresearch/esm:main", "esm2_t36_3B_UR50D"),
    "esm2_15B": partial(load_fn, "facebookresearch/esm:main", "esm2_t48_15B_UR50D"),
}

esm_model_dict = {
    "esm2_8M": {
        "esm_s_dim": 320,
        "esm_z_dim": 120,
        "esm_num_layers": 7,
    },
    "esm2_35M": {
        "esm_s_dim": 480,
        "esm_z_dim": 240,
        "esm_num_layers": 13,
    },
    "esm2_150M": {
        "esm_s_dim": 640,
        "esm_z_dim": 600,
        "esm_num_layers": 31,
    },
    "esm2_650M": {
        "esm_s_dim": 1280,
        "esm_z_dim": 660,
        "esm_num_layers": 34,
    },
    "esm2_3B": {
        "esm_s_dim": 2560,
        "esm_z_dim": 1440,
        "esm_num_layers": 37,
    },
    "esm2_15B": {
        "esm_s_dim": 5120,
        "esm_z_dim": 1920,
        "esm_num_layers": 49,
    },
}

def load_esm(model_key: str = "esm2_150M", device: str | torch.device = "cuda"):
    model, alphabet = esm_registry[model_key]()
    model.eval()
    model.to(device)
    return model, alphabet, model_key

@torch.no_grad()
def calculate_esm_embedding(batch, esm_model, esm_alphabet, layers: list[int]):
    """
    Calculate the ESM embedding for the batch. 
    Inputs:
        -batch: list of feature dictionaries, one dict for each multimer. The feature dictionaries contain the chain dictionaries.
        -esm_model: the ESM model object
        -esm_alphabet: the ESM alphabet object
        -layers: list of integers (layer indices), the layers of the ESM model to output the embeddings from.
    Outputs:
        - the updated batch with the ESM embeddings added to the chain dictionaries as chain_dict["pLM_emb"]=tensor (1, N_r, len(layers), d_e).
    """
    device = next(esm_model.parameters()).device
    sequences = []
    for feat_dict in batch:
        for chain_dict in feat_dict.values():
            sequences.append(chain_dict["sequence"]) # Order: batch x chains flattened
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_batch = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    _, strs, tokens = esm_batch_converter(esm_batch) # tokens: (B_esm, N_max)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = esm_model(tokens, repr_layers=layers, return_contacts=False)
    reps = out["representations"] # dict of layer tensors: {layer_idx: (B_esm, N_max, d_e)}
    embeddings = torch.stack([reps[layer_idx] for layer_idx in layers], dim=-2) # (B_esm, N_max, len(layers), d_e)

    # Build real residue mask:
    pad_idx = esm_alphabet.padding_idx
    cls_idx = esm_alphabet.cls_idx
    eos_idx = esm_alphabet.eos_idx
    valid_mask = ~(
        (tokens == pad_idx) |
        (tokens == cls_idx) |
        (tokens == eos_idx)
    )  # (B_esm, N_max)

    i = 0
    for feat_dict in batch:
        for chain_dict in feat_dict.values():
            chain_dict["pLM_emb"] = embeddings[i][valid_mask[i]].unsqueeze(0) # (1, N_r, len(layers), d_e)
            assert chain_dict["pLM_emb"].size(1) == len(sequences[i]), "Number of residues in the sequence and the embedding do not match."
            assert strs[i] == sequences[i], "Sequence mismatch between ESM output and input sequence."
            i += 1
    
    return batch


########################################################
####                     mint                      #####
########################################################
def calculate_mint_embedding(batch, mint_model, layers: list[int]):
    """
    Calls mint_model and attached the embeddings to the chain dictionaries as chain_dict["pLM_emb"]=tensor (1, N_chain_res + 2, len(layers), d_e).
    Returns the updated batch.
    """

    samples = []
    for feat_dict in batch:
        protein_id = list(feat_dict.values())[0]["protein_id"]
        sequences = [chain_dict["sequence"] for chain_dict in feat_dict.values()]
        sample = Sample(
            id=protein_id,
            chain_id=None,
            sequences=sequences,
            contact_map=None,
            attention_map=None
        )
        samples.append(sample)
    
    collate_fn = CollateFn() # list[Sample, ...] -> ContactPredictionBatch
    mint_batch = collate_fn(samples)

    device = next(mint_model.parameters()).device
    
    with torch.no_grad():
        mint_out = mint_model(
            mint_batch.tokens.to(device),
            mint_batch.chain_ids.to(device),
            need_head_weights=False,
            repr_layers=layers,
        )
    reps = mint_out.hidden_representations # dict of layer tensors: {layer_idx: (1, seq_len, d_e)}
    embeddings = torch.stack([reps[layer_idx] for layer_idx in layers], dim=-2)  # (B, seq_len, len(layers), d_e)

    # Assign embeddings to the chain dictionaries:
    for j, feat_dict in enumerate(batch):
        seq_lens = [len(chain_dict["sequence"]) + 2 for chain_dict in feat_dict.values()] # +2 for the CLS and EOS tokens
        boundaries = [0]
        for L in seq_lens:
            boundaries.append(boundaries[-1] + L)
        
        assert boundaries[-1] == embeddings.size(1), "Number of residues in the sequences and the embeddings do not match."

        for i, chain_dict in enumerate(feat_dict.values()):
            s, e = boundaries[i], boundaries[i+1]
            chain_dict["pLM_emb"] = embeddings[j:j+1, s:e, :, :] # (1, N_r, len(layers), d_e)

            assert chain_dict["pLM_emb"].size(1) == (len(chain_dict["sequence"]) + 2), "Number of residues in the sequence and the embedding do not match."

    return batch    