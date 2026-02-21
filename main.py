# single file for all experiments

import tyro

from configs.process import process_args
from configs.schemas import InputConfig
from obliviator.schemas import TermColor

REPORT_COLOR = TermColor.BRIGHT_MAGENTA
ENCODER_EPOCH = 5
MAX_EPOCHES = 200

cfg = tyro.cli(InputConfig)
eraser, adv_cls, utl_cls, tol = process_args(cfg)

# Initial Accuracy of Unwanted and Utility
print(f"\n{REPORT_COLOR}[Original Accuracy]{TermColor.RESET}")
adv_cls.train(epochs=20)
utl_cls.train(epochs=20)

# Perform Dimensionality Reduction on Data
print(
    f"\n{REPORT_COLOR}[ Dimensionality Reduction | Original Dimesnion:{eraser.x.shape[1]}]{TermColor.RESET}"
)
z, z_test = eraser.null_dim_reduction(tol.dim_reduction)
print()
# Accuracy after Dim reduction
adv_cls.update_input(x=z, x_test=z_test)
adv_cls.train(epochs=MAX_EPOCHES)

utl_cls.update_input(x=z, x_test=z_test)
utl_cls.train(epochs=MAX_EPOCHES)


print(f"\n{REPORT_COLOR}[Starting Iterative Erasure]{TermColor.RESET}\n")
# Initiating the erasure
print(f"{REPORT_COLOR}[Iteration 1]{TermColor.RESET}")
z, z_test = eraser.init_erasure(epochs=ENCODER_EPOCH, tol=tol.evp)
print()
adv_cls.update_input(x=z, x_test=z_test)
adv_cls.train(epochs=MAX_EPOCHES)
utl_cls.update_input(x=z, x_test=z_test)
utl_cls.train(epochs=MAX_EPOCHES)

# Iterative Erasure
for i in range(10):
    print(f"\n{REPORT_COLOR}[Iteration {i + 2}]{TermColor.RESET}")
    z, z_test = eraser.erasure_step(z=z, epochs=ENCODER_EPOCH, tol=tol.evp)
    print()
    adv_cls.update_input(x=z, x_test=z_test)
    adv_cls.train(epochs=MAX_EPOCHES)
    utl_cls.update_input(x=z, x_test=z_test)
    utl_cls.train(epochs=MAX_EPOCHES)
