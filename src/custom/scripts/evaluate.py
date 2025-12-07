"""
Script to evaluate the trained model on the test set.
"""

from pathlib import Path
import hydra
import lightning.pytorch as pl
from omegaconf import DictConfig
from hydra.utils import instantiate

repo_dir = Path(__file__).resolve().parents[2]
config_path = repo_dir / "custom" / "configs"

def evaluate(cfg: DictConfig):
    seed = cfg.get("seed", 57)
    pl.seed_everything(seed, workers=True)

    model = instantiate(cfg.lightning_module)
    test_dataloader = instantiate(cfg.data.test_dataloader)
    trainer = instantiate(cfg.trainer)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path=cfg.paths.model_checkpoint)

@hydra.main(
    version_base="1.3",
    config_path=str(config_path),
    config_name="evaluate.yaml",
)
def main(cfg: DictConfig):
    evaluate(cfg)

if __name__ == "__main__":
    main()