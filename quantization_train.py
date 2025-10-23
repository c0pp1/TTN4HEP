import argparse
from functools import partial
import torch
import numpy as np
from datetime import datetime
from time import perf_counter
import os
import json
import sys
import matplotlib.pyplot as plt
from qtorch import FixedPoint
from qtorch.quant import Quantizer, fixed_point_quantize
from qtorch.optim import OptimLP

from ttnml.ml import TTNModel
from ttnml.tn import check_correct_init
from ttnml.utils import *

from tqdm import tqdm


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


FONTSIZE = 14
slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
slurm_cpus = int(slurm_cpus) if slurm_cpus else 8
torch.set_num_threads(slurm_cpus)
SLURM_ID = os.getenv("SLURM_JOB_ID")

# define json structure
results = {"hyperparams": {"dataset": {}, "model": {}, "training": {}}, "results": {}}

############################
###### SELECT DATASET ######
############################

parser = argparse.ArgumentParser()
parser.add_argument("--bd", type=int, default=10, help="Bond dimension")
parser.add_argument("--nconst", type=int, default=32, help="Number of constituents")
parser.add_argument(
    "--ils",
    type=int,
    nargs="*",
    default=[2],
    help="List of integer lengths. If only one length is provided, it will be used for all fractional lengths.",
)
parser.add_argument(
    "--fls",
    type=int,
    nargs="*",
    default=[2 * i for i in range(1, 9)],
    help="List of fractional lengths.",
)
parser.add_argument(
    "--qgrad",
    action="store_true",
    help="Use quantized gradients.",
)

args = parser.parse_args()

if len(args.ils) == 1:
    FLS = np.array(args.fls, dtype=int)
    WLS = FLS + args.ils[0]
elif len(args.ils) == len(args.fls):
    FLS = np.array(args.fls, dtype=int)
    WLS = FLS + np.array(args.ils)
else:
    raise ValueError("Incompatible lengths for integer and fractional lengths.")

h = 8
n_features = h**2
BATCH_SIZE = 1000
DATASET = "hls150"
MAPPING = "stacked_poly"
MAP_DIM = 2
FEATURES = [4, 5, 13]
TRANSFORM = "log10->5"
NCONST = args.nconst
NORM = "robust"
map_kwargs = (
    {}
)  # {'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}
dataset_params = results["hyperparams"]["dataset"]
dataset_params["name"] = DATASET
dataset_params["batch_size"] = BATCH_SIZE
dataset_params["mapping"] = MAPPING
dataset_params["map_dim"] = MAP_DIM
dataset_params["map_kwargs"] = map_kwargs
dataset_params["nconst"] = NCONST
dataset_params["features"] = FEATURES
dataset_params["norm"] = NORM
dataset_params["transform"] = TRANSFORM

iris_features = ["SL", "SW", "PL", "PW"]

if DATASET == "mnist":
    train_dl, test_dl, train_visual, n_features = get_mnist_data_loaders(
        h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM, labels=[0, 1]
    )
elif DATASET == "stripe":
    train_dl, test_dl, n_features = get_stripeimage_data_loaders(
        4, h, batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "iris":
    # worst performance with iris-versicolor and iris-virginica
    train_dl, test_dl, features = get_iris_data_loaders(
        batch_size=BATCH_SIZE,
        sel_labels=["Iris-setosa", "Iris-virginica", "Iris-versicolor"],
        mapping=MAPPING,
        dim=MAP_DIM,
    )
elif DATASET == "higgs":
    train_dl, test_dl, n_features = get_higgs_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "titanic":
    train_dl, test_dl, n_features = get_titanic_data_loaders(
        batch_size=BATCH_SIZE, scale=(0, 1), mapping=MAPPING, dim=MAP_DIM
    )  # scales different from (0, 1) are reasonable only in the poly mapping
elif DATASET == "bbdata":
    train_dl, test_dl, n_features = get_bb_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=[0, 1, 5, 7, 10, 12, 13, 15],
    )  # permutation=[0,1,5,7,10,12,14,15]
elif DATASET == "hls":
    train_dl, test_dl, n_features = get_hls_data_loaders(
        batch_size=BATCH_SIZE, mapping=MAPPING, dim=MAP_DIM
    )
elif DATASET == "hls150":
    train_dl, test_dl, n_features = get_hls150_data_loaders(
        batch_size=BATCH_SIZE,
        mapping=MAPPING,
        dim=MAP_DIM,
        permutation=FEATURES,
        transform=TRANSFORM,
        nconst=NCONST,
        norm="robust",
    )  # , map_kwargs={'n_part_per_site': 3, 'part_per_feat': [np.arange(12), np.arange(36)]}
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

dataset_params["training_size"] = len(train_dl.dataset)
dataset_params["test_size"] = len(test_dl.dataset)

############################
### SELECT MODEL PARAMS ####
############################

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BOND_DIM = args.bd
N_LABELS = 5
DTYPE = torch.float
dtype_eps = torch.finfo(DTYPE).eps
MODEL_DIR = f"trained_models/{DATASET}_{NORM}_quantmodels/data"
features, n_phys = next(iter(train_dl))[0].shape[-2:]
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_params = results["hyperparams"]["model"]
model_params["bond_dim"] = BOND_DIM
model_params["phys_dim"] = n_phys
model_params["dtype"] = str(DTYPE)
model_params["device"] = DEVICE
model_params["label_dim"] = N_LABELS


##########################
######## TRAINING ########
##########################

LR = 0.001
GAMMA = 0.9
EPOCHS = 100
gauging = False
LOSS_PARAMS = {"l": 1e-4}
SCHEDULER_STEPS = 5
LOSS_FN = class_loss_fn
LOSS = partial(LOSS_FN, **LOSS_PARAMS)
OPTIMIZER = torch.optim.Adam
SCHEDULER = torch.optim.lr_scheduler.ExponentialLR
training_params = results["hyperparams"]["training"]
training_params["optimizer"] = {OPTIMIZER.__name__: {"lr": LR}}
training_params["scheduler"] = {
    SCHEDULER.__name__: {"gamma": GAMMA, "step_size": SCHEDULER_STEPS}
}
training_params["loss"] = {str(LOSS_FN): LOSS_PARAMS}
training_params["epochs"] = EPOCHS
training_params["gauging"] = gauging
training_params["gradient_quantization"] = args.qgrad
training_params["WL"] = WLS.tolist()
training_params["FL"] = FLS.tolist()

now = datetime.now()
results["hyperparams"]["date"] = now.strftime("%Y%m%d-%H%M%S")
folds_train_accs = []
folds_test_accs = []
folds_aucs = []
folds_fprs_at_tpr = []
pbar = tqdm(zip(WLS, FLS), total=len(WLS), desc="QAT", file=sys.stdout, position=0)
for wl, fl in pbar:
    pbar.set_description(f"QAT wl={wl}, fl={fl}")
    tqdm.write(f"\n############# WL {wl}, FL {fl} #############\n")

    forward_num = FixedPoint(wl=wl, fl=fl)
    backward_num = FixedPoint(wl=wl, fl=fl)
    of_suffix = f"wl{wl}_fl{fl}"

    if 2.0 ** (-forward_num.fl) > dtype_eps:
        actual_dtype_eps = 2.0 ** (-forward_num.fl)
    else:
        tqdm.write(
            f"WARNING: Quantization level smaller than dtype eps. Using {DTYPE} type eps instead.",
        )
        actual_dtype_eps = dtype_eps

    # Create a quantizer
    Q = Quantizer(
        forward_number=forward_num,
        backward_number=backward_num,
        forward_rounding="nearest",
        backward_rounding="nearest",
    )

    start = perf_counter()
    model = TTNModel(
        features,
        n_phys=n_phys,
        bond_dim=BOND_DIM,
        n_labels=N_LABELS,
        device=DEVICE,
        dtype=DTYPE,
        quantizer=Q,
    )

    ##########################
    #### INITIALIZE MODEL ####
    ##########################

    INIT_EPOCHS = 5
    loss = lambda *x: class_loss_fn(*x, l=0.01)
    # loss = ClassLoss(0.1, transform=torch.tanh)

    tqdm.write("Initializing the model...", end=" ", file=sys.stdout)
    model.initialize(True, train_dl, loss, INIT_EPOCHS, disable_pbar=True)
    tqdm.write("done \U00002714", file=sys.stdout)
    correct_init, errors = check_correct_init(model, atol=10 * actual_dtype_eps)
    if not correct_init:
        tqdm.write(f"ERROR: Model not correctly initialized. Errors: {errors}")
        continue

    model.to(DEVICE)
    early_stopper = EarlyStopper(patience=10, min_delta=-1e-6)
    optimizer = OPTIMIZER(model.parameters(), lr=LR)
    scheduler = SCHEDULER(optimizer, GAMMA, last_epoch=-1)
    weight_quant = partial(fixed_point_quantize, wl=wl, fl=fl, rounding="nearest")
    acc_quant = partial(
        fixed_point_quantize,
        wl=wl - int(np.floor(np.log2(LR))) + 1,
        fl=fl - int(np.floor(np.log2(LR))) + 1,
        rounding="nearest",
    )

    if args.qgrad:
        tqdm.write("Using quantized gradients.")
        # turn your optimizer into a low precision optimizer
        optimizer = OptimLP(
            optimizer,
            weight_quant=weight_quant,
            grad_quant=weight_quant,
            momentum_quant=acc_quant,
            acc_quant=acc_quant,
        )

    tot_loss_history = []
    mean_epoch_losses = []
    train_accs = []
    test_accs = []
    epoch_pbar = tqdm(
        range(EPOCHS), desc="Training...", total=EPOCHS, file=sys.stdout, position=1
    )
    for epoch in epoch_pbar:
        model.train()
        loss_history = train_one_epoch(
            model, DEVICE, train_dl, LOSS, optimizer, gauging=gauging, disable_pbar=True
        )
        tot_loss_history += loss_history
        mean_epoch_losses.append(np.mean(loss_history))
        epoch_pbar.set_postfix(loss=mean_epoch_losses[-1])

        if early_stopper.early_stop(mean_epoch_losses[-1]):
            tqdm.write(f"Early stopping at epoch {epoch}")
            epoch_pbar.close()
            break

        if epoch % SCHEDULER_STEPS == SCHEDULER_STEPS - 1:
            scheduler.step()
            # pass

        model.eval()
        acc = accuracy(model, DEVICE, train_dl, test_dl, model.dtype, disable_pbar=True)
        train_accs.append(acc[0])
        test_accs.append(acc[1])

    end = perf_counter()
    tqdm.write(f"Training time: {end - start:.2f} seconds")

    loss_history = np.array(tot_loss_history)
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)

    ##########################
    ## EVALUATION AND PLOTS ##
    ##########################

    tqdm.write(f"Train accuracy: {train_accs[-1]}")
    tqdm.write(f"Test accuracy: {test_accs[-1]}")
    tqdm.write(f"Train loss: {loss_history[-1]}")
    folds_train_accs.append(train_accs[-1])
    folds_test_accs.append(test_accs[-1])

    _, fprs_at_tpr, auc, roc_fig, roc_axs = plot_roc_curves(
        model, test_dl, test_accs[-1]
    )
    folds_aucs.append(auc)
    folds_fprs_at_tpr.append(fprs_at_tpr)

    results["results"][f"WL{wl}_FL{fl}"] = {
        "train_accuracy": train_accs[-1],
        "test_accuracy": test_accs[-1],
        "train_loss": loss_history[-1],
        "training_time": end - start,
        "aucs": auc,
        "fprs_at_tpr": fprs_at_tpr,
    }

    accs_fig, accs_ax = plt.subplots(figsize=(6, 4))
    accs_ax.plot(train_accs, label="Train")
    accs_ax.plot(test_accs, label="Test")
    accs_ax.set_xlabel("Epoch", fontsize=FONTSIZE)
    accs_ax.set_ylabel("Accuracy", fontsize=FONTSIZE)
    accs_ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 2)
    accs_ax.legend(fontsize=FONTSIZE)
    accs_fig.tight_layout()

    loss_fig, loss_ax = plt.subplots(1, 1, figsize=(6, 4))
    loss_ax = plot_loss(loss_history, loss_ax, EPOCHS, FS=FONTSIZE)
    loss_ax.set_title(f"Training Loss\n{DATASET}, BD={BOND_DIM}", fontsize=FONTSIZE + 2)
    # ax.set_ylim(1150, 1240)
    loss_fig.tight_layout()

    ######################
    ####### SAVING #######
    ######################

    tqdm.write(f"Saving to {MODEL_DIR}...")
    tqdm.write(
        f"\tLoss history: loss_history_{SLURM_ID}_{of_suffix}.npy",
        end=" ",
    )
    np.save(
        MODEL_DIR + f"/loss_history_{SLURM_ID}_{of_suffix}.npy",
        loss_history,
    )
    tqdm.write("✔")
    tqdm.write(
        f"\tAccuracies: accuracies_{SLURM_ID}_{of_suffix}.npy",
        end=" ",
    )
    np.save(
        MODEL_DIR + f"/accuracies_{SLURM_ID}_{of_suffix}.npy",
        np.stack([train_accs, test_accs], axis=-1),
    )
    tqdm.write("✔")
    tqdm.write(
        f"\tModel: model_{DATASET}_bd{BOND_DIM}_{MAPPING}_{SLURM_ID}_{of_suffix}.npz",
        end=" ",
    )
    model.to_npz(
        MODEL_DIR
        + f"/model_{DATASET}_bd{BOND_DIM}_{MAPPING}_{SLURM_ID}_{of_suffix}.npz"
    )
    tqdm.write("✔")
    tqdm.write("\tPlots:")
    tqdm.write(
        f"\t\tAccuracies: accuracies_{SLURM_ID}_{of_suffix}.pdf",
        end=" ",
    )
    accs_fig.savefig(MODEL_DIR + f"/accuracies_{SLURM_ID}_{of_suffix}.pdf")
    tqdm.write("✔")
    tqdm.write(
        f"\t\tLoss: loss_history_{SLURM_ID}_{of_suffix}.pdf",
        end=" ",
    )
    loss_fig.savefig(MODEL_DIR + f"/loss_history_{SLURM_ID}_{of_suffix}.pdf")
    tqdm.write("✔")
    tqdm.write(
        f"\t\tROCs: roc_curve_{SLURM_ID}_{of_suffix}.pdf",
        end=" ",
    )
    roc_fig.savefig(
        os.path.join(
            MODEL_DIR,
            f"roc_curve_{SLURM_ID}_{of_suffix}.pdf",
        )
    )
    tqdm.write("✔")
    plt.close("all")

results["results"]["train_accs"] = folds_train_accs
results["results"]["test_accs"] = folds_test_accs
results["results"]["aucs"] = folds_aucs
results["results"]["fprs_at_tpr"] = folds_fprs_at_tpr

print("Saving results json...", end=" ")
with open(MODEL_DIR + f"/../results_{SLURM_ID}.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
print("All done ✔")
