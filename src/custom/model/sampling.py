"""
Module contains functionalities for sampling from the model.
"""


import torch
from tqdm import tqdm

from custom.data_processing.data import merge_chains, split_chains


class EulerSampler:
    """
    Euler sampler for ODE.
    """
    
    def __init__(
        self, 
        model,
        num_timesteps=500,
        t_start=1e-4,
        t_end=1.0,
    ):
        self.model = model
        self.num_timesteps = num_timesteps
        self.t_start = t_start
        self.t_end = t_end
        self.timepoints = torch.linspace(t_start, t_end, num_timesteps + 1)

    @torch.no_grad()
    def sample(self, dimer_feat_dict, multiplicity):
        device = next(self.model.parameters()).device
        self.timepoints = self.timepoints.to(device)
        
        # Set initial coordinates:
        x = merge_chains(dimer_feat_dict, "augmented_coords") # (B, N_atoms_A + N_atoms_B, 3). Add initial augmented_coords to the dimer_feat_dict.
        dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x) # Add the initial noised_coords to the dimer_feat_dict

        for i in tqdm(range(self.num_timesteps), desc="Sampling", total=self.num_timesteps):
            t = self.timepoints[i].unsqueeze(0).repeat_interleave(multiplicity, dim=0) # (B,)
            t_next = self.timepoints[i+1].unsqueeze(0).repeat_interleave(multiplicity, dim=0) # (B,)
            dimer_feat_dict = self.model(t, dimer_feat_dict)

            v_predicted = merge_chains(dimer_feat_dict, "velocity_field") # (B, N_atoms_A + N_atoms_B, 3)
            
            delta_x = v_predicted * (t_next - t)
            x = x + delta_x

            # Update coordinates for input to the model in the next time step:
            dimer_feat_dict = split_chains(dimer_feat_dict, "noised_coords", x)


        dimer_feat_dict = split_chains(dimer_feat_dict, "final_sampled_coords", x)
        return dimer_feat_dict