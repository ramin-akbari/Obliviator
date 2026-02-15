# single file for all experiments

import tyro

from configs.schemas import InputConfig

cfg = tyro.cli(InputConfig)
print(cfg)
