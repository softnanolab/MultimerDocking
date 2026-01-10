""" Neural network architecture for protein docking.
    A torch nn.Module which takes in the feature tensors x_t and time step t and
    produces the flow matching outputs (like velocity field and score)."""

import math

import torch
from torch import nn

# SF imports:
from simplefold.model.torch.pos_embed import FourierPositionEncoding, AbsolutePositionEncoding
from simplefold.model.torch.layers import FinalLayer, ConditionEmbedder, TimestepEmbedder


#### - Start: Attention masks - ####
def create_local_attn_bias(
    n: int,
    n_queries: int,
    n_keys: int,
    inf: float = 1e10,
    device: torch.device = None,
) -> torch.Tensor:
    """Create local attention bias based on query window n_queries and kv window n_keys.

    Args:
        n (int): the length of quiries
        n_queries (int): window size of quiries
        n_keys (int): window size of keys/values
        inf (float, optional): the inf to mask attention. Defaults to 1e10.
        device (torch.device, optional): cuda|cpu|None. Defaults to None.

    Returns:
        torch.Tensor: the diagonal-like global attention bias
    """
    n_trunks = int(math.ceil(n / n_queries))
    padded_n = n_trunks * n_queries
    attn_mask = torch.zeros(padded_n, padded_n, device=device)
    for block_index in range(0, n_trunks):
        i = block_index * n_queries
        j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
        j2 = n_queries * block_index + (n_queries + n_keys) // 2
        attn_mask[i : i + n_queries, j1:j2] = 1.0
    attn_bias = (1 - attn_mask) * -inf
    return attn_bias.to(device=device)[:n, :n]

def create_atom_attn_mask(
    device, natoms, atom_n_queries=None, atom_n_keys=None, inf: float = 1e10
) -> torch.Tensor:
    if atom_n_queries is not None and atom_n_keys is not None:
        atom_attn_mask = create_local_attn_bias(
            n=natoms,
            n_queries=atom_n_queries,
            n_keys=atom_n_keys,
            device=device,
            inf=inf,
        )
    else:
        atom_attn_mask = None

    return atom_attn_mask
#### - End: Attention masks - ####


########################################################
####              Encoder architecture             #####
########################################################
class Encoder(nn.Module):
    def __init__(self,
                 d_a,
                 atom_encoder_transformer: nn.Module,
                 atom_n_queries_enc: int = 32,
                 atom_n_keys_enc: int = 128,
                 ):
        super().__init__()
        self.d_a = d_a
        self.atom_n_queries_enc = atom_n_queries_enc
        self.atom_n_keys_enc = atom_n_keys_enc
        self.atom_encoder_transformer = atom_encoder_transformer
        self.E = self.atom_encoder_transformer.blocks[0].hidden_size

        self.pos_embedder = FourierPositionEncoding(
            in_dim=3,
            include_input=True,
            min_freq_log2=0,
            max_freq_log2=12,
            num_freqs=64,
            log_sampling=True,
        )
        PD1 = self.pos_embedder.embed_dim

        self.aminoacid_pos_embedder = AbsolutePositionEncoding(
            in_dim=1,
            embed_dim=self.d_a // 2,
            include_input=True,
        )
        PD2 = self.aminoacid_pos_embedder.embed_dim
        self.L_feat = PD1 + PD2 + 408

        self.atom_feat_proj = nn.Sequential(
            nn.Linear(self.L_feat, self.d_a),
            nn.LayerNorm(self.d_a),
            nn.SiLU(),
        )

        self.atom_pos_proj = nn.Linear(PD1, self.d_a, bias=False)
        self.atom_in_proj = nn.Linear(self.d_a * 2, self.d_a, bias=False)
        self.enc_trunk_proj_in = nn.Sequential(
            nn.Linear(self.d_a, self.E),
            nn.LayerNorm(self.E),
        )
        self.enc_trunk_proj_out = nn.Sequential(
            nn.Linear(self.E, self.d_a),
            nn.LayerNorm(self.d_a),
        )
        self.adaLN_proj = nn.Sequential(
            nn.Linear(self.d_a, self.E),
            nn.LayerNorm(self.E),
        )

    def forward(self,
                atom_coords: torch.Tensor,
                chain_feats: dict,
                adaLN_emb: torch.Tensor,
                RoPE_pos: torch.Tensor,
                atom_to_token: torch.Tensor,
                ) -> dict:
        """
        Note the encoder takes monomer features.
        Inputs:
        - atom_coords: (B, N_atoms, PD1). Positional encoding of the atom coordinates.
        - chain_feats: dict of chain features:
            - add the needed feats here.
        - adaLN_emb: (B, d_a)
        - RoPE_pos: (B, N_atoms, axes)
        - atom_to_token: (B, N_atoms, N_r). 1 if atom belongs to residue, else 0.

        Returns:
        - dict: {"residue_latent": (B, N_r, d_a), 
           "atom_latent": (B, N_atoms, d_a)
           }
        """
        # Build atom_feats:
        ref_pos_emb = self.pos_embedder(chain_feats["ref_pos"]) # (B, N, PD1)
        atom_res_pos = self.aminoacid_pos_embedder(chain_feats["res_id"].float()) # (B, N, 1) -> (B, N, PD2)

        atom_feats = torch.cat(
            [
                ref_pos_emb,  # (B, N, PD1)
                chain_feats["res_type_feat"],  # (B, N, 23)
                atom_res_pos,  # (B, N, PD2)
                chain_feats["charge"],  # (B, N, 1)
                chain_feats["atomic_number"],  # (B, N, 128)
                chain_feats["atom_name_feat"],  # (B, N, 256)
            ],
            dim=-1,
        )  # (B, N, PD1+PD2+408)

        # Preparative projections:
        atom_feats = self.atom_feat_proj(atom_feats) # (B, N_atoms, d_a)

        atom_coords = self.pos_embedder(atom_coords) # (B, N_atoms, PD1)
        atom_coords = self.atom_pos_proj(atom_coords) # (B, N_atoms, d_a)

        atom_in = torch.cat([atom_feats, atom_coords], dim=-1) # (B, N_atoms, 2*d_a)
        atom_in = self.atom_in_proj(atom_in) # (B, N_atoms, d_a)

        atom_latent = self.enc_trunk_proj_in(atom_in) # (B, N_atoms, E)

        adaLN_emb = self.adaLN_proj(adaLN_emb) # (B, E)

        # Encoder transformer:
        attn_mask = create_atom_attn_mask(
            device=atom_coords.device,
            natoms=atom_coords.shape[1],
            atom_n_queries=self.atom_n_queries_enc,
            atom_n_keys=self.atom_n_keys_enc,
        )
        atom_latent = self.atom_encoder_transformer(
            latents=atom_latent,
            c=adaLN_emb,
            attention_mask=attn_mask, # The mask is broadcasted to all heads and all batches. -> Accurate if only one type of protein in the batch.
            pos=RoPE_pos,
        ) # (B, N_atoms, E)
        atom_latent = self.enc_trunk_proj_out(atom_latent) # (B, N_atoms, d_a)

        # Grouping:
        atom_to_token_mean = atom_to_token / (
            atom_to_token.sum(dim=1, keepdim=True) + 1e-6
        )
        residue_latent = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_latent) # (B, N_r, d_a)

        return {"residue_latent": residue_latent, "atom_latent": atom_latent}


########################################################
####            Decoder architecture               #####
########################################################
class Decoder(nn.Module):
    def __init__(self,
                 d_a,
                 D,
                 atom_decoder_transformer: nn.Module,
                 atom_n_queries_dec: int = 32,
                 atom_n_keys_dec: int = 128,
                 ):
        super().__init__()

        self.d_a = d_a
        self.D = D
        self.atom_n_queries_dec = atom_n_queries_dec
        self.atom_n_keys_dec = atom_n_keys_dec

        self.atom_decoder_transformer = atom_decoder_transformer
        
        self.dec_trunk_proj_in = nn.Sequential(
            nn.Linear(self.d_a, self.d_a),
            nn.SiLU(),
            nn.LayerNorm(self.d_a),
            nn.Linear(self.d_a, self.D),
        )
        self.adaLN_proj = nn.Sequential(
            nn.Linear(self.d_a, self.D),
            nn.LayerNorm(self.D),
        )

        self.final_layer = FinalLayer(
            self.D, 3, c_dim=self.d_a
        )
    
    def forward(self,
                residue_latent: torch.Tensor,
                atom_latent: torch.Tensor,
                adaLN_emb: torch.Tensor,
                RoPE_pos: torch.Tensor,
                atom_to_token: torch.Tensor,) -> dict:
        """
        The decoder takes in a sinlge monomeric part of the multimer residue latent and outputs 
        a velocity field for that monomer.
        Inputs:
        - residue_latent: (B, N_r, d_a)
        - atom_latent: (B, N_atoms, d_a). From skip connection around main trunk.
        - adaLN_emb: (B, d_a)
        - RoPE_pos: (B, N_atoms, axes)
        - atom_to_token: (B, N_atoms, N_r). 1 if atom belongs to residue, else 0.
        
        Returns:
        - dict: {"velocity_field": (B, N_atoms, 3),
                 }
        """

        # Ungrouping:
        atom_representation = torch.bmm(atom_to_token, residue_latent) # (B, N_atoms, d_a)

        # Skip connection:
        atom_representation = atom_representation + atom_latent # (B, N_atoms, d_a)

        # Feat. combination and shape preparation:
        atom_representation = self.dec_trunk_proj_in(atom_representation) # (B, N_atoms, D)

        # Decoder transformer:
        adaLN_emb_dec = self.adaLN_proj(adaLN_emb) # (B, D)
        attn_mask = create_atom_attn_mask(
            device=atom_latent.device,
            natoms=atom_latent.shape[1],
            atom_n_queries=self.atom_n_queries_dec,
            atom_n_keys=self.atom_n_keys_dec,
        )
        atom_representation = self.atom_decoder_transformer(
            latents=atom_representation,
            c=adaLN_emb_dec,
            attention_mask=attn_mask, # The mask is broadcasted to all heads and all batches. -> Accurate if only one type of protein in the batch.
            pos=RoPE_pos,
        ) # (B, N_atoms, D)

        velocity_field = self.final_layer(atom_representation, adaLN_emb) # (B, N_atoms, 3)

        return {"velocity_field": velocity_field}
        

########################################################
####            Full model architecture            #####
########################################################
class DockingDiT(nn.Module):
    
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 mtm_res_trunk: nn.Module,
                 pLM_num_layers: int = 31, # for ESM2_150M
                 d_e: int = 640, # for ESM2_150M
                 d_a: int = 768,
                 pLM_dropout_prob: float = 0.0,
                 use_length_condition: bool = True,
                 ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.mtm_res_trunk = mtm_res_trunk
        self.d_e = d_e
        self.d_a = d_a
        self.use_length_condition = use_length_condition

        # pLM:
        self.pLM_combine = nn.Parameter(torch.zeros(pLM_num_layers)) # for multiple layers of pLM embeddings
        self.pLM_proj = ConditionEmbedder(
            input_dim=self.d_e,
            hidden_size=self.d_a,
            dropout_prob=pLM_dropout_prob,
        )
        self.pLM_cat_proj = nn.Linear(2*self.d_a, self.d_a) # For mint

        self.time_embedder = TimestepEmbedder(
            hidden_size=self.d_a,
        )
        if self.use_length_condition:
            self.length_embedder = nn.Sequential(
                nn.Linear(1, self.d_a, bias=False),
                nn.LayerNorm(self.d_a),
            )

        # Learnable SOS and EOS tokens:
        self.seq_start_token = nn.Parameter(torch.randn(1, 1, self.d_a) * 0.02) # (1, 1, d_a)
        self.seq_end_token = nn.Parameter(torch.randn(1, 1,self.d_a) * 0.02) # (1, 1, d_a)


    def forward(self,
                t: torch.Tensor,
                feats: dict,
                ) -> dict:
        """
        Takes in multimer coordinates x_t, time step t and constant (in t) features (including the pLM embedding).
        Outputs velocity field per monomer.

        Inputs:
        - t: (B,). Time step. Same for all chains.

        - feats: dict of feature dicts for each chain:
            - chain_feature_dict[chain_id_1]:
                - noised_coords: tensor (B, N_atoms, 3). Noised coordinates for the current time step t.
                - sequence_length: tensor (B,). Length of the sequence per batch element.
                - res_id: tensor (B, N_atoms, 1). Per atom residue ID.
                - ref_pos: tensor (B, N_atoms, 3). Reference conformer coordinates.
                - atom_to_token: tensor (B, N_atoms, N_r). 1 if atom belongs to residue, else 0.
                - pLM_emb: (1, N_r, len(layers), d_e)
                # TODO: Update the input feature description.
            
            - chain_feature_dict[chain_id_2]:
                - ... (same as for chain 1)

        Returns:
        - dict: {"velocity_fields": {chain_id_1: (B, N_atoms_A, 3),
                                     chain_id_2: (B, N_atoms_B, 3)}}
        """

        # Prepare adaLN time embedding:
        adaLN_emb = self.time_embedder(t) # (B, d_a) 


        ########### Start: Encoding ###########
        chain_residue_latents = []
        chain_atom_latents = []
        chains_RoPE_pos = []
        chains_adaLN_emb = []
        for i, (chain_id, chain_feats) in enumerate(feats.items()):

            # adaLN embedding:
            chain_adaLN_emb = adaLN_emb
            if self.use_length_condition:
                chain_length = chain_feats["sequence_length"] # (B, 1)
                chain_adaLN_emb = adaLN_emb + self.length_embedder(torch.log(chain_length)) # (B, d_a)
            chains_adaLN_emb.append(chain_adaLN_emb)

            # Prepare RoPE:
            chain_RoPE_pos = torch.cat(
            [
            chain_feats["res_id"],
            chain_feats["noised_coords"], # SelfConditioning to keep the structure continuous
            ], dim=-1) # (B, N_atoms, 4)
            chains_RoPE_pos.append(chain_RoPE_pos)

            # Encoding:
            chain_encoding = self.encoder(
                atom_coords=chain_feats["noised_coords"], # (B, N_atoms, 3)
                chain_feats=chain_feats,
                adaLN_emb=chain_adaLN_emb, # (B, d_a)
                RoPE_pos=chain_RoPE_pos, # (B, N_atoms, 4)
                atom_to_token=chain_feats["atom_to_token"], # (B, N_atoms, N_r),
            ) # {"residue_latent": (B, N_r, d_a), "atom_latent": (B, N_atoms, d_a)}

            chain_residue_latents.append(chain_encoding["residue_latent"])
            chain_atom_latents.append(chain_encoding["atom_latent"])
            ########### End: Encoding ###########


        ########### Start: Main trunk ###########
        B = chain_residue_latents[0].shape[0]
        SOS = self.seq_start_token.expand(B, -1, -1) # (B, 1, d_a)
        EOS = self.seq_end_token.expand(B, -1, -1) # (B, 1, d_a)

        chain_A = chain_residue_latents[0] # (B, N_r_A, d_a)
        chain_B = chain_residue_latents[1] # (B, N_r_B, d_a)
        N_r_A = chain_A.shape[1]
        N_r_B = chain_B.shape[1]

        dimer_latent = torch.cat([SOS, chain_A, EOS, SOS, chain_B, EOS], dim=-2) # (B, N_r_A + N_r_B + 4, d_a) = (B, L_r, d_a)

        # Combine with mint embeddings:
        dimer_pLM_emb = torch.cat([chain_feats["pLM_emb"] for chain_feats in feats.values()], dim=1) # (B, N_r_A + N_r_B + 4, pLM_num_layers, d_e) = (B, L_r, num_layers, d_e)
        dimer_pLM_emb = (self.pLM_combine.softmax(0).unsqueeze(0) @ dimer_pLM_emb).squeeze(2) # (B, L_r, d_e)
        dimer_pLM_emb = self.pLM_proj(dimer_pLM_emb, self.training, None) # (B, L_r, d_e) -> (B, L_r, d_a)
        dimer_latent = torch.cat([dimer_latent, dimer_pLM_emb], dim=-1) # (B, L_r, 2*d_a), with each input being (B, L_r, d_a)
        dimer_latent = self.pLM_cat_proj(dimer_latent) # (B, L_r, d_a)


        adaLN_emb_dimer = adaLN_emb
        seq_length = dimer_latent.shape[-2]
        assert seq_length == N_r_A + N_r_B + 4, "Sequence length mismatch"
        if self.use_length_condition:
            seq_length_tensor = torch.tensor(seq_length, device=adaLN_emb.device, dtype=torch.float32).reshape(1, 1) # Same length will be broadcasted to all batches.
            adaLN_emb_dimer = adaLN_emb + self.length_embedder(torch.log(seq_length_tensor)) # (B, d_a)

        mtm_RoPE_pos = torch.arange(seq_length)[None, :, None].expand(B, -1, 1) # (B, N_r_A + N_r_B + 4, 1)

        dimer_latent = self.mtm_res_trunk(
            latents=dimer_latent,
            c=adaLN_emb_dimer,
            attention_mask=None, # The mask would be broadcasted to all heads and all batches. -> Accurate if only one type of protein in the batch.
            pos=mtm_RoPE_pos,
            ) # (B, N_r_A + N_r_B + 4, d_a)
        ########### End: Main trunk ###########


        ########### Start: Decoding ###########
        chain_A = dimer_latent[:, 1:N_r_A+1, :]
        chain_B = dimer_latent[:, N_r_A+3:-1, :]
        assert chain_A.shape[1] == N_r_A and chain_B.shape[1] == N_r_B, "Chain length mismatch"
        chains = [chain_A, chain_B]
        for i, (chain_id, chain_feats) in enumerate(feats.items()):
            decoder_out = self.decoder(
                residue_latent=chains[i],
                atom_latent=chain_atom_latents[i],
                adaLN_emb=chains_adaLN_emb[i],
                RoPE_pos=chains_RoPE_pos[i],
                atom_to_token=chain_feats["atom_to_token"],
            ) # {"velocity_field": (B, N_atoms, 3)}
            chain_feats["velocity_field"] = decoder_out["velocity_field"]
        ########### End: Decoding ###########

        return feats




        


        


