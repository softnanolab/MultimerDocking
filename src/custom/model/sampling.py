"""
Module contains functionalities for sampling from the model.
"""


import torch
from tqdm import tqdm

from custom.data_processing.data import merge_chains, split_chains

from custom.utils.inspect import cif_from_tensor, merge_cifs_into_trajectory


def save_to_trajectory(dimer_feat_dict, i, root_dir, backbone_only, scale_true_coords):
    protein_id = list(dimer_feat_dict.values())[0]["protein_id"]
    file = f"{root_dir}/trajectories/{protein_id}/{i}.cif"
    chain_coords = []
    chain_ids = []
    seqs = []

    for chain_id, chain_dict in dimer_feat_dict.items():
        chain_coords.append(chain_dict["noised_coords"])
        chain_ids.append(chain_id) # IMPORTANT: Biotite only supports 4 letter chain_ids.
        seqs.append(chain_dict["sequence"]) 

    cif_from_tensor(
        chain_coords,
        chain_ids,
        seqs,
        file,
        backbone_only=backbone_only,
        scale=scale_true_coords,
        verbose=False,
    )
    return None


class EulerSampler:
    """
    Euler sampler for ODE.
    """
    
    def __init__(
        self, 
        model,
        backbone_only,
        scale_true_coords,
        num_timesteps=500,
        t_start=1e-4,
        t_end=1.0,
    ):
        self.model = model
        self.backbone_only = backbone_only
        self.scale_true_coords = scale_true_coords
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.t_end = t_end
        self.timepoints = torch.linspace(t_start, t_end, num_timesteps + 1)

    @torch.no_grad()
    def sample_with_plot(self, dimer_feat_dict, multiplicity, trainer_root_dir):
        device = next(self.model.parameters()).device
        self.timepoints = self.timepoints.to(device)
        
        # Set initial coordinates:
        x = merge_chains(dimer_feat_dict, "augmented_coords") # (B, N_atoms_A + N_atoms_B, 3). Add initial augmented_coords to the dimer_feat_dict.
        dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x) # Add the initial noised_coords to the dimer_feat_dict

        #### Save initial structure:
        save_to_trajectory(dimer_feat_dict, 0, trainer_root_dir, self.backbone_only, self.scale_true_coords)
        #### End of saving initial structure

        for i in tqdm(range(self.num_timesteps), desc="Sampling", total=self.num_timesteps):
            t = self.timepoints[i].unsqueeze(0).repeat_interleave(multiplicity, dim=0) # (B,)
            t_next = self.timepoints[i+1].unsqueeze(0).repeat_interleave(multiplicity, dim=0) # (B,)
            dimer_feat_dict = self.model(t, dimer_feat_dict)

            v_predicted = merge_chains(dimer_feat_dict, "velocity_field") # (B, N_atoms_A + N_atoms_B, 3)
            
            dt = t_next - t
            dt = dt[:, None, None] # (B, 1, 1)
            delta_x = v_predicted * dt
            x = x + delta_x

            # Update coordinates for input to the model in the next time step:
            dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x)

            #### Save structure for trajectory visualization:
            save_to_trajectory(dimer_feat_dict, i+1, trainer_root_dir, self.backbone_only, self.scale_true_coords)
            #### End of saving structure

        # Merge cif files into final trajectory:
        protein_id = list(dimer_feat_dict.values())[0]["protein_id"]
        merge_cifs_into_trajectory(f"{trainer_root_dir}/trajectories/{protein_id}", f"{trainer_root_dir}/trajectories/{protein_id}.pdb")


        dimer_feat_dict = split_chains(dimer_feat_dict, "final_sampled_coords", x)
        return dimer_feat_dict