import hydra
from omegaconf import DictConfig

from pref import pref_social_welfare

@hydra.main(version_base=None, config_path="configs", config_name="default")
def my_app(cfg : DictConfig) -> None:
    pref_social_welfare(config=cfg, generator=False)

if __name__ == "__main__":
    my_app()