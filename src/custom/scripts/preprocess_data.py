'''Process the AFDB DDI from raw pdb files into .pt files 
containing coordinates and relevant features.'''


from pathlib import Path
import hydra
from omegaconf import DictConfig

from custom.data_processing.AFDDI_pipeline import process_dataset

repo_dir = Path(__file__).resolve().parents[2]
config_path = repo_dir / "custom" / "configs"

@hydra.main(
    version_base="1.3",
    config_path=str(config_path),
    config_name="preprocess_data",
)

def main(cfg: DictConfig):
    dataset_path = cfg.paths.AFDDI_dataset
    output_path = cfg.paths.AFDDI_output
    N_workers = cfg.N_workers
    process_dataset(dataset_path, output_path, N_workers)

if __name__ == "__main__":
    main()