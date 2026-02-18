# single file for all experiments

import tyro

from configs.process import process_args
from configs.schemas import InputConfig

cfg = tyro.cli(InputConfig)

eraser, adv_cls, utl_cls = process_args(cfg)
# x, x_test = eraser.null_dim_reduction(tol=1e-5)
# print(x.shape)
#
print(adv_cls.net)
print(adv_cls.optimizer)

# adv_cls.update_input(x=x, x_test=x_test)
adv_cls.train(epoch=1500)

# utl_cls.update_input(x=x, x_test=x_test)
utl_cls.train(epoch=1500)
