# single file for all experiments
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

# cs.store(name="base_type",node=)


@hydra.main(version_base=None, config_name="main", config_path="configs")
def main(cfg):

    print(cfg.eraser_encoder.lr)

    print(cfg.prob.lr)


if __name__ == "__main__":
    main()
