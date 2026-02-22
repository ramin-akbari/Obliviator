# single file for all experiments

from typing import Callable

import tyro
from torch import Tensor

from configs.process import process_args
from configs.schemas import InputConfig
from evaluation.probing import MLPCrossEntropy
from obliviator.schemas import TermColor

REPORT_COLOR = TermColor.BRIGHT_MAGENTA
ENCODER_EPOCH = 5
MAX_EPOCHS = 200
MAX_ITER = 15
TARGET_ACC = 0.6


def cls_helper(
    unwanted_cls: MLPCrossEntropy, utility_cls: MLPCrossEntropy
) -> tuple[Callable[[Tensor, Tensor], None], Callable[[int], float]]:
    def helper_update(z: Tensor, z_test: Tensor) -> None:
        unwanted_cls.update_input(x=z, x_test=z_test)
        utility_cls.update_input(x=z, x_test=z_test)

    def helper_accuracy(epochs: int) -> float:
        unwanted_cls.train(epochs)
        utility_cls.train(epochs)
        return unwanted_cls.max_acc

    return helper_update, helper_accuracy


def main():
    cfg = tyro.cli(InputConfig)
    eraser, adv_cls, utl_cls, tol = process_args(cfg)
    update_cls, update_accuracy = cls_helper(adv_cls, utl_cls)

    # Initial Accuracy of Unwanted and Utility
    print(f"\n{REPORT_COLOR}[Original Accuracy]{TermColor.RESET}")
    update_accuracy(25)

    # Perform Dimensionality Reduction on Data
    print(
        f"\n{REPORT_COLOR}[ Dimensionality Reduction | Original Dimension:{eraser.x.shape[1]}]{TermColor.RESET}"
    )
    z, z_test = eraser.null_dim_reduction(tol.dim_reduction)
    print()
    # Accuracy after Dim reduction
    update_cls(z, z_test)
    update_accuracy(MAX_EPOCHS)

    print(f"\n{REPORT_COLOR}[Starting Iterative Erasure]{TermColor.RESET}\n")
    print(f"{REPORT_COLOR}[Iteration 1]{TermColor.RESET}")

    # Initiating the erasure
    z, z_test = eraser.init_erasure(epochs=ENCODER_EPOCH, tol=tol.evp)
    print()
    update_cls(z, z_test)
    unwanted_acc = update_accuracy(MAX_EPOCHS)
    it = 1

    # Iterative Erasure
    while unwanted_acc > TARGET_ACC and it < MAX_ITER:
        it += 1
        print(f"\n{REPORT_COLOR}[Iteration {it}]{TermColor.RESET}")
        z, z_test = eraser.erasure_step(z=z, epochs=ENCODER_EPOCH, tol=tol.evp)
        print()
        update_cls(z, z_test)
        update_accuracy(MAX_EPOCHS)


if __name__ == "__main__":
    main()
