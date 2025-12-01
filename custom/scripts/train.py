"""
Train the docking model.
"""

import sys
from pathlib import Path

# Add the repo directory to Python path so custom package can be imported
repo_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_dir))

import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.utils import instantiate


def train(cfg: DictConfig):
    seed = cfg.get("seed", 57)
    pl.seed_everything(seed, workers=True)
    
    model = instantiate(cfg.lightning_module)
    dataloader = instantiate(cfg.data.dataloader)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloaders=dataloader)

@hydra.main(
    version_base="1.3",
    config_path="/rds/general/user/emb25/home/dock_proj/repo/custom/configs",
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
