# single file for all experiments

import tyro

from configs.process import process_args
from configs.schemas import InputConfig

cfg = tyro.cli(InputConfig)

oblv_config, probing_config, probing_optimizer, data = process_args(cfg)

print(data)
