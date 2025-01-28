import argparse
import torch
import numpy as np
import logging
from datetime import datetime
from ttn_torch import TTNModel
from utils import *
import sys
import os

from tqdm import tqdm, trange

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from images.plot_utils import *

logger = logging.getLogger(__name__)


def parse_arguments(cmd):

    parser = argparse.ArgumentParser(description="Train a TTN model on a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
        choices=["mnist", "stripe", "iris", "bbdata", "titanic", "hls"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train the model."
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for training."
    )
    parser.add_argument(
        "--sweeps",
        type=int,
        default=20,
        help="Number of sweeps to train the model. If 0 falls back to SGD.",
    )
    parser.add_argument(
        "--max-bond", type=int, default=10, help="Maximum bond dimension for the TTN."
    )
    parser.add_argument(
        "--dtype", type=str, default="float64", help="Data type for the model."
    )
    parser.add_argument(
        "--emb-map", type=str, default="spin", help="Embedding map for the TTN."
    )
    parser.add_argument(
        "--map-dim", type=int, default=2, help="Dimension of the embedding map."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Relative output filename for the model.",
    )
    parser.add_argument(
        "--img-size", type=int, default=4, help="Size of the images in the dataset."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for training."
    )

    args = parser.parse_args(cmd)
    return args


def load_data(args):
    dataset = args.dataset
    h = args.img_size
    MAPPING = args.emb_map
    MAP_DIM = args.map_dim
    BATCH_SIZE = args.batch_size

    if dataset == "mnist":
        train_dl, test_dl, _, features = get_mnist_data_loaders(
            h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM, labels=[0, 1]
        )
    elif dataset == "stripe":
        train_dl, test_dl, features = get_stripeimage_data_loaders(
            4, h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
    elif dataset == "iris":
        # worst performance with iris-versicolor and iris-virginica
        train_dl, test_dl, features = get_iris_data_loaders(
            batch_size=BATCH_SIZE,
            sel_labels=["Iris-setosa", "Iris-virginica", "Iris-versicolor"],
            mapping=MAPPING,
            dim=MAP_DIM,
        )
    elif dataset == "higgs":
        train_dl, test_dl, features = get_higgs_data_loaders(
            batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
    elif dataset == "titanic":
        train_dl, test_dl, features = get_titanic_data_loaders(
            batch_size=BATCH_SIZE, scale=(0, 1), mapping=MAPPING, dim=MAP_DIM
        )  # scales different from (0, 1) are reasonable only in the poly mapping
    elif dataset == "bbdata":
        train_dl, test_dl, features = get_bb_data_loaders(
            batch_size=BATCH_SIZE,
            mapping=MAPPING,
            dim=MAP_DIM,
            permutation=[0, 1, 5, 7, 10, 12, 13, 15],
        )  # permutation=[0,1,5,7,10,12,14,15]
    elif dataset == "hls":
        train_dl, test_dl, features = get_hls_data_loaders(
            batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_dl, test_dl, features


def train_model(args, train_dl, test_dl, features):

    EPOCHS = args.epochs
    MAX_BOND = args.max_bond
    DTYPE = getattr(torch, args.dtype)
    DATASET = args.dataset
    MAP_DIM = args.map_dim
    OUTPUT = args.output
    SWEEPS = args.sweeps
    DEVICE = args.device
    EMB_MAP = args.emb_map
    INIT_EPOCHS = 15
    N_LABELS = next(iter(train_dl))[1].shape[-1]

    if OUTPUT == "":
        OUTPUT = f"{args.dataset}_models/"

    # Initialize the model
    model = TTNModel(
        features,
        n_phys=MAP_DIM,
        bond_dim=MAX_BOND,
        n_labels=N_LABELS,
        device=DEVICE,
        dtype=DTYPE,
    )

    init_loss = lambda *x: class_loss_fn(*x, l=0.01)
    model.initialize(True, train_dl, init_loss, INIT_EPOCHS)

    now = datetime.now()
    # Train the model
    if SWEEPS > 0:
        loss = lambda *x: class_loss_fn(*x, l=0.0)
        optimizer = torch.optim.Adam(
            model.parameters(), 2 ** (model.n_layers - 6)
        )  # 2**(model.n_layers-6)

        loss_history = []
        for s in trange(SWEEPS, desc="sweeps", position=0):
            losses, _ = model.sweep(
                train_dl,
                dclass_loss_fn,
                optimizer,
                epochs=EPOCHS,
                path_type="layer+0",
                manual=True,
                loss=loss,
                verbose=1,
                save_grads=False,
            )
            loss_history.extend(losses)

        loss_history = np.array(loss_history)
    else:
        gauging = False
        SCHEDULER_STEPS = 5
        LOSS = lambda *x: class_loss_fn(*x, l=0.01)

        model.train()
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=2 ** (model.n_layers - 6))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 0.9, last_epoch=-1, verbose=False
        )

        tot_loss_history = []
        for epoch in trange(EPOCHS, desc="epochs", position=0):
            loss_history = train_one_epoch(
                model, DEVICE, train_dl, LOSS, optimizer, gauging=gauging
            )
            tot_loss_history += loss_history

            if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS - 1:
                scheduler.step()
                # pass

        loss_history = np.array(tot_loss_history)

    train_acc, test_acc = accuracy(model, DEVICE, train_dl, test_dl, model.dtype)

    logger.info(f"Training time: {datetime.now() - now}")
    logger.info(f"Final loss: {loss_history[-1]}")
    logger.info(f"Final train/test accuracy: {train_acc:.3f}/{test_acc:.3f}")

    # Save the model
    model.to_npz(
        os.path.join(
            OUTPUT,
            f"model_{DATASET}_bd{MAX_BOND}_{EMB_MAP}_{now.strftime('%Y%m%d-%H%M%S')}.npz",
        )
    )
    np.save(
        os.path.join(
            OUTPUT,
            f"model_{DATASET}_bd{MAX_BOND}_{EMB_MAP}_{now.strftime('%Y%m%d-%H%M%S')}_loss.npy",
        ),
        loss_history,
    )

    return model, loss_history


def main(cmd):
    args = parse_arguments(cmd)

    logger.info("Loading data...")
    train_dl, test_dl, features = load_data(args)

    if args.output == "":
        args.output = f"{args.dataset}_models/"

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logger.info("Training model...")
    model, loss_history = train_model(args, train_dl, test_dl, features)
    logger.info("Training complete.")

    return


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    main(sys.argv[1:])
