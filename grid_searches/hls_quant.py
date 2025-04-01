# This script trains 15 models changing their quantization.
# The quantization is changed keeping the number of bits for the integer part fixed to 2
# and changing the number of bits for the fractional part from 2 to 16 (i.e. changing the word length).
# The models are trainedon the HLS dataset using the SGD optimisation.
# The models are saved in the folder ../trained_models/hls_quant.

# pylint: disable=invalid-name

from functools import partial
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from qtorch import FixedPoint
from qtorch.quant import Quantizer, fixed_point_quantize
from qtorch.optim import OptimLP
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, trange

module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(module_dir, ".."))

from ttnml.ml import TTNModel
from ttnml.tn import check_correct_init
from ttnml.utils import *

logger = logging.getLogger(__name__)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def main():

    BATCH_SIZE = 1000
    MAPPING = "spin"
    MAP_DIM = 2

    # Load the dataset
    logger.info("Loading the dataset")
    train_dl, test_dl, features = get_hls_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )

    # Define the model structure
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BOND_DIM = 10
    N_LABELS = 5
    MODEL_DIR = os.path.join(module_dir, "..", "trained_models/hls_quant")
    DTYPE = (
        torch.float
    )  # we use float for the quantization, as qtorch does not support double
    dtype_eps = torch.finfo(DTYPE).eps

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # save this for later
    logger.info("Saving the dataset")
    x_train = torch.cat([x for x, _ in train_dl], dim=0).numpy()
    x_test = torch.cat([x for x, _ in test_dl], dim=0).numpy()
    y_train_true = (
        torch.cat([y for _, y in train_dl], dim=0).to(dtype=torch.int).numpy()
    )
    y_test_true = torch.cat([y for _, y in test_dl], dim=0).to(dtype=torch.int).numpy()
    np.save(os.path.join(MODEL_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(MODEL_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(MODEL_DIR, "y_train_true.npy"), y_train_true)
    np.save(os.path.join(MODEL_DIR, "y_test_true.npy"), y_test_true)

    # Define some training Hyperparameters
    logger.info("Preparing the training")
    INIT_EPOCHS = 5
    init_loss = partial(class_loss_fn, l=0.01)
    LR = 0.01
    EPOCHS = 50
    gauging = False
    LAMBDA = 1e-4
    training_loss = partial(class_loss_fn, l=LAMBDA)

    FLs = np.arange(2, 17)
    df = pd.DataFrame(columns=["wl", "fl", "train_acc", "test_acc", "auc"])
    df["fl"] = FLs
    wls = []
    train_accs = []
    test_accs = []
    aucs = []

    pbar = tqdm(FLs, desc="Quantization search", position=10)
    for fl in pbar:
        pbar.set_postfix({"fl": fl})
        # Define the quantization
        wl = 2 + fl
        forward_num = FixedPoint(wl=wl, fl=fl)
        backward_num = FixedPoint(wl=wl, fl=fl)

        of_prefix = f"wl{wl}_fl{fl}"

        if 2.0 ** (-forward_num.fl) > dtype_eps:
            actual_dtype_eps = 2.0 ** (-forward_num.fl)
        else:
            logger.warning(
                "FL: %s - Quantization level smaller than dtype eps. Using %s type eps instead.",
                fl,
                DTYPE,
            )
            actual_dtype_eps = dtype_eps

        # Create a quantizer
        Q = Quantizer(
            forward_number=forward_num,
            backward_number=backward_num,
            forward_rounding="nearest",
            backward_rounding="nearest",
        )

        model = TTNModel(
            features,
            bond_dim=BOND_DIM,
            n_labels=N_LABELS,
            device=DEVICE,
            dtype=DTYPE,
            quantizer=Q,
        )

        logger.info("FL: %s - Initializing...", fl)
        model.initialize(
            True,
            train_dl,
            init_loss,
            INIT_EPOCHS,
            position=1,
            leave=False,
            disable_pbar=True,
        )

        correct_init, errors = check_correct_init(model, atol=actual_dtype_eps)
        if not correct_init:
            logger.error(
                "Model with fl %s not correctly initialized. Errors: %s", fl, errors
            )
            continue

        model.train()
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        if model.quantizer is not None:
            # define custom quantization functions for different numbers
            weight_quant = partial(
                fixed_point_quantize, wl=wl, fl=fl, rounding="nearest"
            )
            # turn your optimizer into a low precision optimizer
            optimizer = OptimLP(optimizer, weight_quant=weight_quant)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.75, patience=5, min_lr=1e-4
        )
        early_stopper = EarlyStopper(patience=10, min_delta=-1e-6)

        tot_loss_history = []
        mean_epoch_losses = []
        epochs_pbar = trange(EPOCHS, desc=f"FL: {fl} - Training...", position=1)
        last_lr = LR
        for epoch in epochs_pbar:
            loss_history = train_one_epoch(
                model,
                DEVICE,
                train_dl,
                training_loss,
                optimizer,
                gauging=gauging,
                disable_pbar=True,
            )
            tot_loss_history += loss_history
            mean_epoch_losses.append(np.mean(loss_history))
            epochs_pbar.set_postfix({"loss": mean_epoch_losses[-1]})

            if early_stopper.early_stop(mean_epoch_losses[-1]):
                logger.info("FL: %s - Early stopping at epoch %s", fl, epoch)
                epochs_pbar.close()
                break

            scheduler.step(mean_epoch_losses[-1])
            current_lr = scheduler.get_last_lr()[0]
            if current_lr < last_lr:
                logger.info(
                    "FL: %s - Lowering the lr from %s to %s", fl, last_lr, current_lr
                )
                last_lr = current_lr

        loss_history = np.array(tot_loss_history)

        train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, model.dtype)
        logger.info("FL: %s - Test accuracy: %s", fl, test_acc)
        test_pred = get_predictions(model, DEVICE, test_dl)
        auc = roc_auc_score(y_test_true, test_pred, multi_class="ovr", average="micro")

        np.save(os.path.join(MODEL_DIR, of_prefix + "-loss_history.npy"), loss_history)
        train_out = get_output(model, DEVICE, train_dl)
        test_out = get_output(model, DEVICE, test_dl)
        np.save(os.path.join(MODEL_DIR, of_prefix + "-train_out.npy"), train_out)
        np.save(os.path.join(MODEL_DIR, of_prefix + "-test_out.npy"), test_out)

        model.to_npz(os.path.join(MODEL_DIR, of_prefix + "-model.npz"))

        wls.append(wl)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        aucs.append(auc)

    df["wl"] = wls
    df["train_acc"] = train_accs
    df["test_acc"] = test_accs
    df["auc"] = aucs
    df.to_csv(os.path.join(MODEL_DIR, "results.csv"))

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
