# single file for all experiments

import tyro

from configs.process import process_args
from configs.schemas import InputConfig

cfg = tyro.cli(InputConfig)

eraser, adv_cls, utl_cls, tol = process_args(cfg)

# Initial Accuracy of Unwanted and Utility
adv_cls.train(epochs=20)
utl_cls.train(epochs=20)

# Perform Dimensionality Reduction on Data
x, x_test = eraser.null_dim_reduction(tol.dim_reduction)

# Accuracy after Dim reduction
adv_cls.update_input(x=x, x_test=x_test)
adv_cls.train(epochs=80)

utl_cls.update_input(x=x, x_test=x_test)
utl_cls.train(epochs=80)
