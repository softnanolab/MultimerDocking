"""
Train the docking model.
"""
from pathlib import Path
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

repo_dir = Path(__file__).resolve().parents[2]
config_path = repo_dir / "custom" / "configs"

def train(cfg: DictConfig):
    seed = cfg.get("seed", 57)
    pl.seed_everything(seed, workers=True)
    
    model = instantiate(cfg.lightning_module)
    dataloader = instantiate(cfg.data.dataloader)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, train_dataloaders=dataloader)

@hydra.main(
    version_base="1.3",
    config_path=str(config_path),
    config_name="training.yaml",
)
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()
