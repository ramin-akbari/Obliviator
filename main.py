# single file for all experiments

import tyro

from configs.process import process_args
from configs.schemas import InputConfig

cfg = tyro.cli(InputConfig)

eraser, adv_cls, utl_cls = process_args(cfg)

print(eraser)
