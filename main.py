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
z, z_test = eraser.null_dim_reduction(tol.dim_reduction)


# Accuracy after Dim reduction
adv_cls.update_input(x=z, x_test=z_test)
adv_cls.train(epochs=100)

utl_cls.update_input(x=z, x_test=z_test)
utl_cls.train(epochs=100)


# # Starting Erasure
z, z_test = eraser.init_erasure(epochs=5, tol=tol.evp)

# # Iterative Erasure
for i in range(10):
    adv_cls.update_input(x=z, x_test=z_test)
    adv_cls.train(epochs=80)

    utl_cls.update_input(x=z, x_test=z_test)
    utl_cls.train(epochs=80)

    x, x_test = eraser.erasure_step(z=z, epochs=5, tol=tol.evp)
